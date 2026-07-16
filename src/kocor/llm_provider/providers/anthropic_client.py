"""Anthropic LLM 客户端实现。

使用 anthropic SDK，负责内部 Message 格式与 Anthropic API 格式之间的转换。
"""

from __future__ import annotations

import json

from anthropic import Anthropic

from kocor._secret import SecretStr
from kocor.llm_provider.llm_client import BaseLLMClient
from kocor.llm_provider.message import FunctionCall, Message, StreamChunk, ToolCall, Usage
from kocor.tools.definitions import ToolDefinition


def _reveal(key: SecretStr | str) -> str:
    """兼容 SecretStr 和普通 str 的 API Key 读取。"""
    return key.reveal() if isinstance(key, SecretStr) else key


class AnthropicClient(BaseLLMClient):
    """Anthropic LLM 客户端。"""

    def _create_client(self):
        return Anthropic(
            api_key=_reveal(self.config.anthropic_api_key),
            auth_token=_reveal(self.config.anthropic_api_key),  # anthropic 兼容不同厂商模型
            base_url=self.config.anthropic_base_url or None,
        )

    @property
    def provider(self) -> str:
        return "anthropic"

    def _prepare_messages(self, messages: list[Message]) -> tuple:
        """从消息中提取 system 消息。

        Anthropic 将 system 消息作为顶层参数传入，不支持在 messages 数组中
        包含 role=system 的消息。多个 system 消息用分隔符拼接。
        """
        system_parts: list[str] = []
        filtered_messages = []
        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
            else:
                filtered_messages.append(msg)
        # Anthropic API 的 system 参数是单独传递的顶层字段，不在 messages 数组中
        system_content = "\n\n---\n\n".join(system_parts)
        return (system_content or None), filtered_messages

    def _api_generate(self, messages_data, tools_data, max_tokens, temperature, system=None):
        """调用 Anthropic API 生成响应（非流式）。"""
        return self._client.messages.create(
            model=self.config.anthropic_model,
            system=system,
            messages=messages_data,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools_data,
        )

    def _api_stream(self, messages_data, tools_data, max_tokens, temperature, system=None):
        """流式调用 Anthropic API 生成响应。"""
        import httpx

        # 流式场景需要短 read timeout（3s）以保证 Windows 上 Ctrl+C 的响应性。
        # 因为此超时与 generate() 场景（需要更长 read timeout）不同，所以使用
        # 独立的 SDK 客户端创建，不共享 __init__ 中的 self._client。
        client = Anthropic(
            api_key=_reveal(self.config.anthropic_api_key),
            auth_token=_reveal(self.config.anthropic_api_key),  # anthropic 兼容不同厂商模型
            base_url=self.config.anthropic_base_url or None,
            # 设置 read timeout 确保 blocking socket read 能定期返回 Python 字节码，
            # 从而让 Windows 上的 KeyboardInterrupt(Ctrl+C) 可以被传递。
            # 用 httpx.Timeout 而非简单 float 以精确控制每个阶段的超时。
            timeout=httpx.Timeout(
                connect=30.0,  # TCP 连接建立（通常 <1s）
                read=3.0,      # 两次数据块之间的最大等待（Ctrl+C 最多等 3s）
                write=30.0,    # 请求体发送
                pool=30.0,     # 连接池等待
            ),
        )

        accumulated_text = ""
        accumulated_tool_calls: dict[int, ToolCall] = {}
        tool_block_starts: dict[int, dict] = {}  # index → block metadata
        prompt_tokens = 0
        cached_tokens = 0

        for event in client.messages.create(
            model=self.config.anthropic_model,
            system=system,
            messages=messages_data,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools_data,
            stream=True,
        ):
            stream_chunk = StreamChunk()

            # Anthropic 流式事件的类型包括：
            # message_start → content_block_start → content_block_delta* → content_block_stop → message_delta
            match event.type:
                case "message_start":
                    # 第一个事件，携带 prompt 侧的 token 用量
                    usage_attr = getattr(event, "message", None)
                    if usage_attr and hasattr(usage_attr, "usage"):
                        prompt_tokens = getattr(usage_attr.usage, "input_tokens", 0)
                        cached_tokens = getattr(usage_attr.usage, "cache_read_input_tokens", 0)

                case "content_block_delta":
                    # 增量更新：文本增量、JSON 增量（工具参数）、思维链增量
                    if event.delta.type == "text_delta":
                        text = event.delta.text
                        accumulated_text += text
                        stream_chunk.content = text
                    elif event.delta.type == "input_json_delta":
                        idx = event.index
                        json_fragment = event.delta.partial_json
                        if idx not in accumulated_tool_calls:
                            # 第一个 JSON 增量——从 content_block_start 获取骨架（工具名和 ID）
                            block_meta = tool_block_starts.get(idx, {})
                            accumulated_tool_calls[idx] = ToolCall(
                                id=block_meta.get("id", ""),
                                type="function",
                                function=FunctionCall(
                                    name=block_meta.get("name", ""),
                                    arguments=json_fragment,
                                ),
                            )
                        else:
                            # 后续 JSON 增量——追加到 arguments
                            accumulated_tool_calls[idx].function.arguments += json_fragment
                        stream_chunk.tool_calls = list(accumulated_tool_calls.values())
                    elif event.delta.type == "thinking_delta":
                        stream_chunk.reasoning = event.delta.thinking

                case "content_block_stop":
                    # 工具块结束，yield 一次完整 tool_calls
                    if event.index in accumulated_tool_calls:
                        stream_chunk.tool_calls = list(accumulated_tool_calls.values())

                case "content_block_start":
                    # 记录工具块开始信息（名称和 ID），供后续 JSON delta 使用
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        tool_block_starts[event.index] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                        }

                case "message_delta":
                    # 最后事件，携带 completion 侧的 token 用量和 stop_reason
                    if event.delta.stop_reason:
                        stream_chunk.is_final = True
                    usage_attr = getattr(event, "usage", None)
                    if usage_attr:
                        output_tokens = getattr(usage_attr, "output_tokens", 0)
                        stream_chunk.usage = Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=prompt_tokens + output_tokens,
                            cached_tokens=cached_tokens,
                        )

            # 有内容时才 yield——避免向渲染层发无意义的空 chunk
            if stream_chunk.content or stream_chunk.reasoning or stream_chunk.tool_calls or stream_chunk.is_final:
                yield stream_chunk

    # ── 格式转换 ──

    def _normalize_in(self, messages: list[Message]) -> list[dict]:
        """内部消息格式 → Anthropic 消息格式

        Anthropic 要求 tool_result 类型消息必须以 content 块数组形式作为 user 消息发送。
        如果连续出现多条 tool 消息（对应多次工具调用），它们会合并为同一条 user 消息的 content 数组。
        """
        result: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush_tool_results():
            """将累积的 tool_results 作为一条 user 消息写入 result。"""
            nonlocal pending_tool_results
            if pending_tool_results:
                result.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for msg in messages:
            if msg.role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                })
                continue

            # 非 tool 消息，先 flush 累积的 tool_results
            _flush_tool_results()

            match msg.role:
                case "user":
                    result.append({"role": "user", "content": msg.content})
                case "assistant":
                    if msg.tool_calls:
                        # assistant 消息带工具调用时，content 必须为块数组格式
                        content_blocks = []
                        if msg.content:
                            content_blocks.append({"type": "text", "text": msg.content})
                        for tc in msg.tool_calls:
                            try:
                                input_dict = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                input_dict = {}
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": input_dict,
                            })
                        result.append({"role": "assistant", "content": content_blocks})
                    else:
                        result.append({"role": "assistant", "content": msg.content})

        # 末尾可能还有未 flush 的 tool_results
        _flush_tool_results()
        return result

    def _normalize_out(self, response, usage: Usage | None = None) -> Message:
        """Anthropic response 格式 → 内部消息格式"""
        content_blocks = response.content
        prompt = getattr(response.usage, "input_tokens", 0)
        completion = getattr(response.usage, "output_tokens", 0)
        cached = getattr(response.usage, "cache_read_input_tokens", 0)
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cached_tokens=cached,
        )

        # 提取 thinking（思维链）
        reasoning = ""
        for block in content_blocks:
            if block.type == "thinking":
                reasoning += block.thinking

        # 检查是否有工具调用
        tool_blocks = [b for b in content_blocks if b.type == "tool_use"]
        if tool_blocks:
            tool_calls = []
            for block in tool_blocks:
                input_data = block.input
                # dict → JSON 字符串
                if isinstance(input_data, dict):
                    arguments = json.dumps(input_data)
                else:
                    arguments = str(input_data)
                tool_calls.append(ToolCall(
                    id=block.id,
                    function=FunctionCall(
                        name=block.name,
                        arguments=arguments,
                    ),
                ))
            return Message(
                role="assistant",
                content="",
                reasoning=reasoning,
                tool_calls=tool_calls,
                usage=usage,
            )

        # 纯文本响应
        text_blocks = [b for b in content_blocks if b.type == "text"]
        content = " ".join(b.text for b in text_blocks) if text_blocks else ""
        return Message(role="assistant", content=content, reasoning=reasoning, usage=usage)

    def _normalize_tool(self, tool: ToolDefinition) -> dict:
        """内部工具定义 → Anthropic 工具格式

        Anthropic 的工具 Schema 字段名为 input_schema（而非 OpenAI 的 parameters）。
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }