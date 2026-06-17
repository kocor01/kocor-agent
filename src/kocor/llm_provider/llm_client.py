"""LLM 客户端抽象层。

提供统一的 LLMClient 接口，具体实现由 OpenAI/Anthropic SDK 完成。
只做格式归一化，不封装 LLM 的能力差异。
"""

from __future__ import annotations

from typing import Iterator, Protocol

from kocor.config import Config
from kocor.llm_provider.tool_definition import ToolDefinition
from kocor.message import Message, StreamChunk


class LLMClient(Protocol):
    """LLM 客户端抽象接口。

    所有 provider 实现必须遵循此接口。
    """

    @property
    def provider(self) -> str:
        """返回 provider 名称: 'openai' | 'anthropic'"""
        ...

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Message:
        """生成响应。

        Args:
            messages: 消息列表（含历史）
            tools: 可用工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Returns:
            Message:
            - 纯文本: Message(role="assistant", content="...")
            - 工具调用: Message(role="assistant", content="", tool_calls=[...])
        """
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式生成响应。

        Args:
            messages: 消息列表（含历史）
            tools: 可用工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Yields:
            StreamChunk: 流式数据块
        """
        ...