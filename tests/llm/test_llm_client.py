"""测试 LLMClient 抽象接口"""


from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message, StreamChunk
from kocor.tools.definitions import ToolDefinition


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
        from kocor.llm_provider.message import FunctionCall, ToolCall

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
