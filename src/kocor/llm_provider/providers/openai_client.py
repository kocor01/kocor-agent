"""OpenAI LLM 客户端实现。

使用 openai SDK，负责内部 Message 格式与 OpenAI API 格式之间的转换。
"""

from __future__ import annotations

from openai import OpenAI

from kocor._secret import SecretStr
from kocor.llm_provider.llm_client import BaseLLMClient
from kocor.llm_provider.message import FunctionCall, Message, StreamChunk, ToolCall, Usage
from kocor.tools.definitions import ToolDefinition


def _reveal(key: SecretStr | str) -> str:
    """兼容 SecretStr 和普通 str 的 API Key 读取。"""
    return key.reveal() if isinstance(key, SecretStr) else key


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM 客户端。"""

    def _create_client(self):
        """创建 OpenAI SDK 客户端（懒加载，首次调用时创建）。"""
        return None  # 懒加载：首次 _get_client() 时创建

    def _get_client(self) -> OpenAI:
        """获取或创建 SDK 客户端实例（懒加载，复用连接池）。"""
        if self._client is None:
            self._client = OpenAI(
                api_key=_reveal(self.config.openai_api_key),
                base_url=self.config.openai_base_url or None,
            )
        return self._client

    @property
    def provider(self) -> str:
        """返回提供商标识 "openai"。"""
        return "openai"

    def _api_generate(self, messages_data, tools_data, max_tokens, temperature, system=None):
        """调用 OpenAI API 生成响应（非流式）。"""
        return self._get_client().chat.completions.create(
            model=self.config.openai_model,
            messages=messages_data,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools_data,
        )

    def _api_stream(self, messages_data, tools_data, max_tokens, temperature, system=None):
        """流式调用 OpenAI API 生成响应。"""
        import httpx

        # 流式场景需要短 read timeout（3s）以保证 Windows 上 Ctrl+C 的响应性。
        # 因为此超时与 generate() 不同，使用独立的 SDK 客户端。
        client = OpenAI(
            api_key=_reveal(self.config.openai_api_key),
            base_url=self.config.openai_base_url or None,
            # 设置 read timeout 确保 blocking socket read 能定期返回 Python 字节码，
            # 从而让 Windows 上的 KeyboardInterrupt(Ctrl+C) 可以被传递。
            timeout=httpx.Timeout(
                connect=30.0,
                read=3.0,      # Ctrl+C 最多等 3s
                write=30.0,
                pool=30.0,
            ),
        )

        accumulated_tool_calls: dict[int, ToolCall] = {}

        for chunk in client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages_data,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools_data,
            stream=True,
            stream_options={"include_usage": True},
        ):
            # 捕获 token 用量（独立 chunk，无 choices，不 yield）
            if not chunk.choices and chunk.usage:
                usage_chunk = StreamChunk(
                    is_final=True,
                    usage=Usage(
                        prompt_tokens=getattr(chunk.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(chunk.usage, "completion_tokens", 0),
                        total_tokens=getattr(chunk.usage, "total_tokens", 0),
                        cached_tokens=(
                            getattr(chunk.usage, "prompt_tokens_details", None)
                            and getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                            or 0
                        ),
                    ),
                )
                yield usage_chunk
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            is_final = finish_reason is not None and finish_reason != ""

            # reasoning 增量（兼容 OpenAI o-series reasoning 和 DeepSeek reasoning_content）
            reasoning_text = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)

            # 构建 chunk（reasoning 传增量，与 content 一致）
            stream_chunk = StreamChunk(
                content=delta.content or "",
                reasoning=reasoning_text or "",
                is_final=is_final,
            )

            # 累积 tool_calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = ToolCall(
                            id=tc_delta.id or "",
                            type=tc_delta.type or "function",
                            function=FunctionCall(
                                name=tc_delta.function.name or "",
                                arguments=tc_delta.function.arguments or "",
                            ),
                        )
                    else:
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx].function.arguments += tc_delta.function.arguments
                stream_chunk.tool_calls = list(accumulated_tool_calls.values())

            yield stream_chunk

    # ── 格式转换 ──

    def _normalize_in(self, messages: list[Message]) -> list[dict]:
        """内部消息格式 → OpenAI 消息格式"""
        result = []
        for msg in messages:
            match msg.role:
                case "system":
                    result.append({"role": "system", "content": msg.content})
                case "user":
                    result.append({"role": "user", "content": msg.content})
                case "assistant":
                    body: dict = {"role": "assistant"}
                    if msg.content:
                        body["content"] = msg.content
                    if msg.reasoning:
                        body["reasoning"] = msg.reasoning
                    if msg.tool_calls:
                        tool_calls = []
                        for tc in msg.tool_calls:
                            tool_calls.append({
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            })
                        body["tool_calls"] = tool_calls
                    result.append(body)
                case "tool":
                    result.append({
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    })
        return result

    def _normalize_out(self, response, usage: Usage | None = None) -> Message:
        """OpenAI response 格式 → 内部消息格式

        Args:
            response: chat.completions.create 的完整响应对象
            usage: 可选的用量信息，None 时从 response 提取
        """
        choice = response.choices[0]
        message = choice.message

        if usage is None:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            total_tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
            cached_tokens = 0
            if (
                response.usage
                and hasattr(response.usage, "prompt_tokens_details")
                and response.usage.prompt_tokens_details
            ):
                cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
            )

        # 检查是否有工具调用
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                ))
            return Message(
                role="assistant",
                content=message.content or "",
                reasoning=getattr(message, "reasoning", None) or "",
                tool_calls=tool_calls,
                usage=usage,
            )

        return Message(
            role="assistant",
            content=message.content or "",
            reasoning=getattr(message, "reasoning", None) or "",
            usage=usage,
        )

    def _normalize_tool(self, tool: ToolDefinition) -> dict:
        """内部工具定义 → OpenAI API 工具格式"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }