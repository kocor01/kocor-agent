"""测试 LLMClient 抽象接口"""


from kocor.config import LLMConfig
from kocor.llm_client import LLMClient, ToolDefinition, create_llm_client, register_client
from kocor.message import Message, StreamChunk


class TestLLMClientInterface:
    """测试 LLMClient 接口契约"""

    def test_interface_is_protocol(self):
        """LLMClient 必须是 Protocol"""
        assert isinstance(LLMClient, type)
        # Protocol 有 __protocol_attrs__
        assert hasattr(LLMClient, "__protocol_attrs__")

    def test_interface_has_generate(self):
        """接口必须有 generate 方法"""
        assert "generate" in LLMClient.__protocol_attrs__

    def test_interface_has_provider_property(self):
        """接口必须有 provider 属性"""
        assert "provider" in LLMClient.__protocol_attrs__

    def test_generate_returns_message(self):
        """generate 返回 Message 类型"""
        # 用一个简单的 mock 验证
        class FakeClient(LLMClient):
            @property
            def provider(self) -> str:
                return "fake"

            def generate(
                self,
                messages: list[Message],
                tools: list[ToolDefinition] | None = None,
                max_tokens: int = 4096,
                temperature: float = 0.0,
            ) -> Message:
                return Message(role="assistant", content="hello")

        client = FakeClient()
        result = client.generate([Message(role="user", content="hi")])
        assert isinstance(result, Message)
        assert result.content == "hello"

    def test_generate_with_tools_returns_message_with_tool_calls(self):
        """工具调用时返回含 tool_calls 的 Message"""
        from kocor.message import FunctionCall, ToolCall

        class FakeClient(LLMClient):
            @property
            def provider(self) -> str:
                return "fake"

            def generate(
                self,
                messages: list[Message],
                tools: list[ToolDefinition] | None = None,
                max_tokens: int = 4096,
                temperature: float = 0.0,
            ) -> Message:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                        )
                    ],
                )

        client = FakeClient()
        result = client.generate([Message(role="user", content="read a.txt")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "read_file"


class TestLLMClientStream:
    """测试 LLMClient stream 方法"""

    def test_interface_has_stream(self):
        """接口必须有 stream 方法"""
        assert "stream" in LLMClient.__protocol_attrs__

    def test_stream_returns_iterator(self):
        """stream 返回 Iterator[StreamChunk]"""

        class FakeClient(LLMClient):
            @property
            def provider(self) -> str:
                return "fake"

            def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0) -> Message:
                return Message(role="assistant", content="hello")

            def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                yield StreamChunk(content="hello", is_final=True)

        client = FakeClient()
        chunks = list(client.stream([Message(role="user", content="hi")]))
        assert len(chunks) == 1
        assert chunks[0].content == "hello"
        assert chunks[0].is_final is True


class TestCreateLLMClient:
    """测试 LLMClient 工厂函数"""

    def test_create_openai_client(self):
        """测试创建 OpenAI 客户端"""
        from kocor.openai_client import OpenAIClient

        config = LLMConfig(provider="openai")
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)
        assert client.provider == "openai"

    def test_create_anthropic_client(self):
        """测试创建 Anthropic 客户端"""
        from kocor.anthropic_client import AnthropicClient

        config = LLMConfig(provider="anthropic")
        client = create_llm_client(config)
        assert isinstance(client, AnthropicClient)
        assert client.provider == "anthropic"

    def test_create_unsupported_provider(self):
        """测试不支持的 provider 抛出异常"""
        config = LLMConfig(provider="unknown")
        try:
            create_llm_client(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "不支持的 provider" in str(e)

    def test_register_client(self):
        """测试 register_client 注册新 provider"""
        class FakeClient(LLMClient):
            def __init__(self, config: LLMConfig = None):
                pass

            @property
            def provider(self) -> str:
                return "fake"

            def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0) -> Message:
                return Message(role="assistant", content="fake")

        register_client("fake", FakeClient)
        try:
            config = LLMConfig(provider="fake")
            client = create_llm_client(config)
            assert isinstance(client, FakeClient)
            assert client.provider == "fake"
        finally:
            # 清理注册表
            from kocor.llm_client import _clients
            _clients.pop("fake", None)
