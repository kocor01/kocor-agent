"""LLM 客户端工厂与注册。"""

from __future__ import annotations

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient


class LlmManager:
    """管理 LLM 客户端的注册与创建。"""

    _client: LLMClient | None = None
    _clients: dict[str, type[LLMClient]] = {}

    @classmethod
    def register(cls, provider: str, client_class: type[LLMClient]) -> None:
        """注册 LLM 客户端实现。

        Args:
            provider: provider 名称
            client_class: 客户端类（必须实现 LLMClient 协议）
        """
        cls._clients[provider] = client_class

    @classmethod
    def create(cls) -> LLMClient:
        """根据系统配置创建 LLM 客户端。

        Returns:
            对应的 LLMClient 实例

        Raises:
            ValueError: 不支持的 provider
        """
        if not cls._clients:
            from kocor.llm_provider.providers import AnthropicClient, OpenAIClient

            cls.register("openai", OpenAIClient)
            cls.register("anthropic", AnthropicClient)

        provider = Config.get("provider")
        client_class = cls._clients.get(provider)
        if client_class is None:
            raise ValueError(
                f"不支持的 provider: '{provider}'，"
                f"可选值: {sorted(cls._clients.keys())}"
            )
        return client_class()

    @classmethod
    def reset(cls) -> None:
        """重置缓存和注册表（用于测试）。"""
        cls._client = None
        cls._clients.clear()

    @classmethod
    def get_llm_client(cls) -> LLMClient:
        """获取 LLM 客户端实例（单例）。

        首次调用时根据系统配置创建并缓存，后续返回已缓存的实例。

        Returns:
            对应的 LLMClient 实例

        Raises:
            ValueError: 不支持的 provider
        """
        if cls._client is not None:
            return cls._client
        cls._client = cls.create()
        return cls._client


