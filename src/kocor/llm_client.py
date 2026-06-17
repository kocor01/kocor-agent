"""LLM 客户端工厂与注册。

从 base_client 导入 LLMClient/ToolDefinition，提供注册和工厂函数。
"""

from __future__ import annotations

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient


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
        from kocor.llm_provider.anthropic_client import AnthropicClient
        from kocor.llm_provider.openai_client import OpenAIClient

        register_client("openai", OpenAIClient)
        register_client("anthropic", AnthropicClient)

    client_class = _clients.get(config.provider)
    if client_class is None:
        raise ValueError(
            f"不支持的 provider: '{config.provider}'，"
            f"可选值: {sorted(_clients.keys())}"
        )
    return client_class(config)
