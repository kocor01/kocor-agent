"""LLM 客户端抽象层。

提供统一的 LLMClient 接口，具体实现由 OpenAI/Anthropic SDK 完成。
只做格式归一化，不封装 LLM 的能力差异。
"""

from __future__ import annotations

from typing import Iterator, Protocol

from kocor.config import Config
from kocor.message import Message, StreamChunk


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


_clients: dict[str, type[LLMClient]] = {}


def register_client(provider: str, client_class: type[LLMClient]) -> None:
    """注册 LLM 客户端实现。

    Args:
        provider: provider 名称
        client_class: 客户端类（必须实现 LLMClient 协议）
    """
    _clients[provider] = client_class


def create_llm_client(config: Config) -> LLMClient:
    """根据配置创建 LLM 客户端。

    Args:
        config: LLM 配置对象

    Returns:
        对应的 LLMClient 实例

    Raises:
        ValueError: 不支持的 provider
    """
    # 内置注册（惰性，仅在首次调用时执行）
    if not _clients:
        from kocor.anthropic_client import AnthropicClient
        from kocor.openai_client import OpenAIClient

        register_client("openai", OpenAIClient)
        register_client("anthropic", AnthropicClient)

    client_class = _clients.get(config.provider)
    if client_class is None:
        raise ValueError(
            f"不支持的 provider: '{config.provider}'，"
            f"可选值: {sorted(_clients.keys())}"
        )
    return client_class(config)
