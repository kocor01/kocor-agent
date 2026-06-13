"""LLM 客户端抽象层。

提供统一的 LLMClient 接口，具体实现由 OpenAI/Anthropic SDK 完成。
只做格式归一化，不封装 LLM 的能力差异。
"""

from __future__ import annotations

from typing import Protocol

from kocor.config import LLMConfig
from kocor.message import Message, ToolCall


class ToolDefinition:
    """工具定义，用于 JSON Schema 描述。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 参数定义
    """

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict:
        """转换为 OpenAI API 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


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


def create_llm_client(config: LLMConfig) -> LLMClient:
    """根据配置创建 LLM 客户端。

    Args:
        config: LLM 配置对象

    Returns:
        对应的 LLMClient 实例

    Raises:
        ValueError: 不支持的 provider
    """
    match config.provider:
        case "openai":
            from kocor.openai_client import OpenAIClient

            return OpenAIClient(config)
        case "anthropic":
            from kocor.anthropic_client import AnthropicClient

            return AnthropicClient(config)
        case unknown:
            raise ValueError(f"不支持的 provider: {unknown}")
