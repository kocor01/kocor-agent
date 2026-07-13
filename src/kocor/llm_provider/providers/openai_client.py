"""OpenAI LLM 客户端实现。

使用 openai SDK，负责内部 Message 格式与 OpenAI API 格式之间的转换。
"""

from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.tools.definitions import ToolDefinition
from kocor.llm_provider.message import FunctionCall, Message, StreamChunk, ToolCall, Usage


class OpenAIClient(LLMClient):
    """OpenAI LLM 客户端。"""

    def __init__(self):
        self.config = Config.load()

    @property
    def provider(self) -> str:
        return "openai"

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Message:
        """调用 OpenAI API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度（默认使用 Config.max_tokens）
            temperature: 采样温度

        Returns:
            Message: 响应消息
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url or None,
        )

        # 内部格式 → OpenAI 格式
        openai_messages = self._normalize_in(messages)
        openai_tools = [self._to_openai_tool(t) for t in tools] if tools else None

        # 调用 API
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=openai_messages,
            max_tokens=actual_max_tokens,
            temperature=temperature,
            tools=openai_tools,
        )

        # OpenAI 格式 → 内部格式
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
        total_tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
        cached_tokens = 0
        if response.usage and hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
            cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )
        return self._normalize_out(response.choices[0], usage=usage)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式调用 OpenAI API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度（默认使用 Config.max_tokens）
            temperature: 采样温度

        Yields:
            StreamChunk: 流式数据块
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        import httpx

        client = OpenAI(
            api_key=self.config.openai_api_key,
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

        openai_messages = self._normalize_in(messages)
        openai_tools = [self._to_openai_tool(t) for t in tools] if tools else None

        accumulated_tool_calls: dict[int, ToolCall] = {}

        try:
            for chunk in client.chat.completions.create(
                model=self.config.openai_model,
                messages=openai_messages,
                max_tokens=actual_max_tokens,
                temperature=temperature,
                tools=openai_tools,
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

        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            # 同 anthropic_client.py：Windows 上 blocking socket read 阻止 KeyboardInterrupt，
            # 用 read timeout 突破阻塞后跳过超时异常，让上层 KeyboardInterrupt 处理逻辑生效。
            raise KeyboardInterrupt()

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

    def _normalize_out(self, choice, usage: Usage | None = None) -> Message:
        """OpenAI choice 格式 → 内部消息格式"""
        message = choice.message

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

    @staticmethod
    def _to_openai_tool(tool: ToolDefinition) -> dict:
        """ToolDefinition → OpenAI API 工具格式"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
