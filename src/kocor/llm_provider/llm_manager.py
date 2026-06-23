"""LLM 客户端工厂与注册。"""

from __future__ import annotations

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient


class LlmManager:
    """管理 LLM 客户端的注册与创建。"""

    _instance: LlmManager | None = None
    _clients: dict[str, type[LLMClient]] = {}

    @classmethod
    def get_instance(cls) -> LlmManager:
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, provider: str, client_class: type[LLMClient]) -> None:
        """注册 LLM 客户端实现。

        Args:
            provider: provider 名称
            client_class: 客户端类（必须实现 LLMClient 协议）
        """
        self._clients[provider] = client_class

    def create(self, config: Config) -> LLMClient:
        """根据配置创建 LLM 客户端。

        Args:
            config: LLM 配置对象

        Returns:
            对应的 LLMClient 实例

        Raises:
            ValueError: 不支持的 provider
        """
        # 内置注册（惰性，仅在首次调用时执行）
        if not self._clients:
            from kocor.llm_provider.providers import AnthropicClient, OpenAIClient

            self.register("openai", OpenAIClient)
            self.register("anthropic", AnthropicClient)

        client_class = self._clients.get(config.provider)
        if client_class is None:
            raise ValueError(
                f"不支持的 provider: '{config.provider}'，"
                f"可选值: {sorted(self._clients.keys())}"
            )
        return client_class(config)

    @classmethod
    def reset(cls) -> None:
        """重置单例和注册表（用于测试）。"""
        cls._instance = None
        cls._clients.clear()

    @classmethod
    def register_client(cls, provider: str, client_class: type[LLMClient]) -> None:
        """注册 LLM 客户端实现。

        Args:
            provider: provider 名称
            client_class: 客户端类（必须实现 LLMClient 协议）
        """
        cls.get_instance().register(provider, client_class)

    @classmethod
    def create_llm_client(cls, config: Config) -> LLMClient:
        """根据配置创建 LLM 客户端。

        Args:
            config: LLM 配置对象

        Returns:
            对应的 LLMClient 实例

        Raises:
            ValueError: 不支持的 provider
        """
        return cls.get_instance().create(config)


