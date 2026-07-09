"""Anthropic LLM 客户端实现。

使用 anthropic SDK，负责内部 Message 格式与 Anthropic API 格式之间的转换。
"""

from __future__ import annotations

import json
from typing import Iterator

from anthropic import Anthropic

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.tools.definitions import ToolDefinition
from kocor.llm_provider.message import FunctionCall, Message, StreamChunk, ToolCall, Usage


class AnthropicClient(LLMClient):
    """Anthropic LLM 客户端。"""

    def __init__(self):
        self.config = Config.load()

    @property
    def provider(self) -> str:
        return "anthropic"

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Message:
        """调用 Anthropic API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度（默认使用 Config.max_tokens）
            temperature: 采样温度

        Returns:
            Message: 响应消息
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        client = Anthropic(
            api_key=self.config.anthropic_api_key,
            auth_token=self.config.anthropic_api_key,  # anthropic 兼容不同厂商模型
            base_url=self.config.anthropic_base_url or None,
        )

        system_content, filtered_messages = self._extract_system(messages)
        anthropic_messages = self._normalize_in(filtered_messages)
        anthropic_tools = [self._normalize_tool(t) for t in tools] if tools else None

        # 调用 API
        response = client.messages.create(
            model=self.config.anthropic_model,
            system=system_content,
            messages=anthropic_messages,
            max_tokens=actual_max_tokens,
            temperature=temperature,
            tools=anthropic_tools,
        )

        # Anthropic 格式 → 内部格式
        return self._normalize_out(response)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式调用 Anthropic API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度（默认使用 Config.max_tokens）
            temperature: 采样温度

        Yields:
            StreamChunk: 流式数据块
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        client = Anthropic(
            api_key=self.config.anthropic_api_key,
            auth_token=self.config.anthropic_api_key,  # anthropic 兼容不同厂商模型
            base_url=self.config.anthropic_base_url or None,
        )

        system_content, filtered_messages = self._extract_system(messages)
        anthropic_messages = self._normalize_in(filtered_messages)
        anthropic_tools = [self._normalize_tool(t) for t in tools] if tools else None

        accumulated_text = ""
        accumulated_tool_calls: dict[int, ToolCall] = {}
        tool_block_starts: dict[int, dict] = {}  # index → block metadata
        prompt_tokens = 0
        cached_tokens = 0

        for event in client.messages.create(
            model=self.config.anthropic_model,
            system=system_content,
            messages=anthropic_messages,
            max_tokens=actual_max_tokens,
            temperature=temperature,
            tools=anthropic_tools,
            stream=True,
        ):
            stream_chunk = StreamChunk()

            match event.type:
                case "message_start":
                    usage_attr = getattr(event, "message", None)
                    if usage_attr and hasattr(usage_attr, "usage"):
                        prompt_tokens = getattr(usage_attr.usage, "input_tokens", 0)
                        cached_tokens = getattr(usage_attr.usage, "cache_read_input_tokens", 0)

                case "content_block_delta":
                    if event.delta.type == "text_delta":
                        text = event.delta.text
                        accumulated_text += text
                        stream_chunk.content = text
                    elif event.delta.type == "input_json_delta":
                        idx = event.index
                        json_fragment = event.delta.partial_json
                        if idx not in accumulated_tool_calls:
                            # 从 content_block_start 获取骨架
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
                            accumulated_tool_calls[idx].function.arguments += json_fragment
                        stream_chunk.tool_calls = list(accumulated_tool_calls.values())
                    elif event.delta.type == "thinking_delta":
                        stream_chunk.reasoning = event.delta.thinking

                case "content_block_stop":
                    # 工具块结束，yield 一次完整 tool_calls
                    if event.index in accumulated_tool_calls:
                        stream_chunk.tool_calls = list(accumulated_tool_calls.values())

                case "content_block_start":
                    # 记录工具块开始信息
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        tool_block_starts[event.index] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                        }

                case "message_delta":
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

            # 有内容时才 yield
            if stream_chunk.content or stream_chunk.reasoning or stream_chunk.tool_calls or stream_chunk.is_final:
                yield stream_chunk

    # ── 格式转换 ──

    @staticmethod
    def _extract_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """从消息列表中提取 system 消息并过滤。

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
        system_content = "\n\n---\n\n".join(system_parts)
        return system_content or None, filtered_messages

    def _normalize_in(self, messages: list[Message]) -> list[dict]:
        """内部消息格式 → Anthropic 消息格式"""
        result: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush_tool_results():
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

    def _normalize_out(self, response) -> Message:
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
        """内部工具定义 → Anthropic 工具格式"""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
