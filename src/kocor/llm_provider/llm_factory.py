"""LLM 客户端工厂——纯工厂，无状态缓存。

每次调用 create() 返回新实例，不持有任何客户端引用。
"""

from __future__ import annotations

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient


class LlmFactory:
    """LLM 客户端工厂，每次调用 create() 返回新实例。

    不缓存客户端实例，调用方负责管理客户端生命周期。
    注册表 _providers 在模块级别共享（类变量），但仅用于查找 provider 类，不存储实例状态。
    """

    _providers: dict[str, type[LLMClient]] = {}

    @classmethod
    def register(cls, provider: str, client_class: type[LLMClient]) -> None:
        """注册 LLM 客户端实现。

        Args:
            provider: provider 名称（如 "openai"、"anthropic"）
            client_class: 客户端类（必须实现 LLMClient 协议）
        """
        cls._providers[provider] = client_class

    @classmethod
    def create(cls) -> LLMClient:
        """根据系统配置创建 LLM 客户端。

        首次调用时自动注册内置的 openai 和 anthropic 客户端。

        Returns:
            对应的 LLMClient 实例

        Raises:
            ValueError: 不支持的 provider 或没有注册任何 provider
        """
        if not cls._providers:
            cls._register_builtins()

        provider = Config.get("provider")
        client_class = cls._providers.get(provider)
        if client_class is None:
            raise ValueError(
                f"不支持的 provider: '{provider}'，"
                f"可选值: {sorted(cls._providers.keys())}"
            )
        return client_class()

    @classmethod
    def _register_builtins(cls) -> None:
        """注册内置的 openai 和 anthropic 客户端实现。"""
        from kocor.llm_provider.providers import AnthropicClient, OpenAIClient

        cls.register("openai", OpenAIClient)
        cls.register("anthropic", AnthropicClient)