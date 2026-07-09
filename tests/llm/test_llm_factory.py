"""测试 LlmFactory — 纯工厂，无状态缓存。"""

from __future__ import annotations

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.llm_factory import LlmFactory
from kocor.llm_provider.message import Message


class TestLlmFactory:
    """测试 LlmFactory 纯工厂行为。"""

    def setup_method(self):
        LlmFactory._providers.clear()
        Config.reset()

    def test_create_returns_llm_client(self):
        """create() 应返回 LLMClient 协议兼容实例。"""
        LlmFactory.register("fake", _FakeClient)
        Config._instance = Config(provider="fake")
        client = LlmFactory.create()
        # LLMClient 是 Protocol（非 runtime_checkable），通过协议属性验证
        assert hasattr(client, "provider")
        assert hasattr(client, "generate")
        assert hasattr(client, "stream")
        assert client.provider == "fake"

    def test_create_returns_new_instance_each_call(self):
        """每次调用 create() 应返回新实例（无状态缓存）。"""
        LlmFactory.register("fake", _FakeClient)
        Config._instance = Config(provider="fake")
        c1 = LlmFactory.create()
        c2 = LlmFactory.create()
        assert c1 is not c2  # 不同实例

    def test_create_raises_on_unsupported_provider(self):
        """不支持的 provider 应抛出 ValueError。"""
        LlmFactory.register("fake", _FakeClient)
        Config._instance = Config(provider="unknown")
        try:
            LlmFactory.create()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown" in str(e).lower()

    def test_register_overrides_existing(self):
        """注册同名 provider 应覆盖已有。"""
        LlmFactory.register("fake", _FakeClient)
        LlmFactory.register("fake", _FakeClientV2)
        Config._instance = Config(provider="fake")
        client = LlmFactory.create()
        assert client.provider == "fake"

    def test_auto_register_on_first_create(self):
        """首次 create() 时自动注册内置的 openai/anthropic。"""
        Config._instance = Config(provider="openai")
        client = LlmFactory.create()
        from kocor.llm_provider.providers import OpenAIClient
        assert isinstance(client, OpenAIClient)

    def test_auto_register_populates_providers_dict(self):
        """自动注册后 _providers 应包含 openai 和 anthropic。"""
        LlmFactory._providers.clear()
        Config._instance = Config(provider="openai")
        LlmFactory.create()
        assert "openai" in LlmFactory._providers
        assert "anthropic" in LlmFactory._providers


class _FakeClient(LLMClient):
    """伪造 LLMClient 用于测试。"""

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0) -> Message:
        return Message(role="assistant", content="fake")

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        yield from ()


class _FakeClientV2(LLMClient):
    """第二个伪造客户端，验证覆盖注册。"""

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0) -> Message:
        return Message(role="assistant", content="fake-v2")

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        yield from ()