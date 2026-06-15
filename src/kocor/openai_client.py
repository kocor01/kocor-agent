"""OpenAI LLM 客户端实现。

使用 openai SDK，负责内部 Message 格式与 OpenAI API 格式之间的转换。
"""

from __future__ import annotations

import os
from typing import Iterator

from openai import OpenAI

from kocor.config import LLMConfig
from kocor.llm_client import LLMClient, ToolDefinition
from kocor.message import FunctionCall, Message, StreamChunk, ToolCall


class OpenAIClient(LLMClient):
    """OpenAI LLM 客户端。

    Attributes:
        config: LLM 配置
        _model: 模型名称（从环境变量读取）
        _base_url: 兼容端点（从环境变量读取）
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._model = os.environ.get("OPENAI_MODEL", "GPT-5.5")
        self._base_url = os.environ.get("OPENAI_BASE_URL") or None

    @property
    def provider(self) -> str:
        return "openai"

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Message:
        """调用 OpenAI API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Returns:
            Message: 响应消息
        """
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        # 内部格式 → OpenAI 格式
        openai_messages = self._normalize_in(messages)
        openai_tools = [t.to_dict() for t in tools] if tools else None

        # 调用 API
        response = client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=openai_tools,
        )

        # OpenAI 格式 → 内部格式
        return self._normalize_out(response.choices[0])

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式调用 OpenAI API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Yields:
            StreamChunk: 流式数据块
        """
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        openai_messages = self._normalize_in(messages)
        openai_tools = [t.to_dict() for t in tools] if tools else None

        accumulated_tool_calls: dict[int, ToolCall] = {}

        for chunk in client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=openai_tools,
            stream=True,
        ):
            delta = chunk.choices[0].delta
            is_final = chunk.choices[0].finish_reason is not None

            # reasoning 增量（仅推理模型返回此字段）
            reasoning_text = getattr(delta, "reasoning", None)

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

    def _normalize_out(self, choice) -> Message:
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
            )

        return Message(
            role="assistant",
            content=message.content or "",
            reasoning=getattr(message, "reasoning", None) or "",
        )
