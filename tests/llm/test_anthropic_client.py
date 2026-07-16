"""测试 Anthropic 客户端"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from kocor.config import Config
from kocor.llm_provider.message import FunctionCall, Message, ToolCall
from kocor.llm_provider.providers import AnthropicClient
from kocor.tools.definitions import ToolDefinition


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolBlock:
    type: str = "tool_use"
    id: str = "toolu_1"
    name: str = "read_file"
    input: dict = None  # type: ignore

    def __post_init__(self):
        if self.input is None:
            self.input = {}


class TestAnthropicClient:
    """测试 Anthropic 客户端"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="anthropic")

    def test_provider(self):
        client = AnthropicClient()
        assert client.provider == "anthropic"

    
    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_generate_text_response(self, mock_anthropic_cls):
        """测试纯文本响应"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="你好，我是助手")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="你好")])

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "你好，我是助手"
        assert result.tool_calls == []

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_generate_tool_call_response(self, mock_anthropic_cls):
        """测试工具调用响应"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_tool_block = MockToolBlock(
            id="toolu_123",
            name="read_file",
            input={"path": "test.txt"},
        )
        mock_response = MagicMock(content=[mock_tool_block], stop_reason="tool_use")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="读 test.txt")])

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].function.name == "read_file"
        import json
        assert json.loads(result.tool_calls[0].function.arguments) == {"path": "test.txt"}

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_generate_with_tools(self, mock_anthropic_cls):
        """测试传入工具定义"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="我来读文件")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        tools = [
            ToolDefinition(
                name="read_file",
                description="读文件",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        client.generate([Message(role="user", content="读文件")], tools=tools)

        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_generate_uses_config_max_tokens_default(self, mock_anthropic_cls):
        """不传 max_tokens 时使用 Config 的 max_tokens 值"""
        Config._instance = Config(provider="anthropic", max_tokens=2048)
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock(
            content=[MockTextBlock(text="hello")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        client.generate([Message(role="user", content="hi")])

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 2048

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_stream_uses_config_max_tokens_default(self, mock_anthropic_cls):
        """流式调用时不传 max_tokens 使用 Config 的 max_tokens 值"""
        Config._instance = Config(provider="anthropic", max_tokens=1024)
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.return_value = [
            MockContentBlockDelta(delta=MockTextDelta(text="hi")),
            MockMessageDelta(_stop_reason="end_turn"),
            MockMessageStop(),
        ]

        client = AnthropicClient()
        list(client.stream([Message(role="user", content="hi")]))

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 1024

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_generate_system_message(self, mock_anthropic_cls):
        """测试 system 消息传递"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="hello")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="hi"),
        ]
        client.generate(messages)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["system"] == "你是助手"

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_generate_multiple_system_messages(self, mock_anthropic_cls):
        """多个 system 消息应拼接而非覆盖。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="done")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="忽略"),
            Message(role="system", content="[历史摘要] 用户问了天气"),
            Message(role="assistant", content="晴天"),
        ]
        client.generate(messages)

        call_args = mock_client.messages.create.call_args
        system = call_args.kwargs["system"]
        assert "你是助手" in system
        assert "历史摘要" in system
        # 过滤后的消息不应包含 system
        msgs = call_args.kwargs["messages"]
        assert all(m["role"] != "system" for m in msgs)

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_normalize_tool_result(self, mock_anthropic_cls):
        """测试工具结果格式归一化"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MockTextBlock(text="done")],
            stop_reason="end_turn",
        )

        client = AnthropicClient()
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="toolu_1", function=FunctionCall(name="read_file", arguments='{}')),
            ]),
            Message(role="tool", content="file content", tool_call_id="toolu_1"),
        ]
        result = client.generate(messages)
        assert result.content == "done"

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_normalize_multiple_tool_results(self, mock_anthropic_cls):
        """测试多个 tool_result 被合并到同一条 user 消息"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MockTextBlock(text="merged")],
            stop_reason="end_turn",
        )

        client = AnthropicClient()
        messages = [
            Message(role="user", content="do two things"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="toolu_1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
                ToolCall(id="toolu_2", function=FunctionCall(name="write_file", arguments='{"path":"b.txt"}')),
            ]),
            Message(role="tool", content="content a", tool_call_id="toolu_1"),
            Message(role="tool", content="content b", tool_call_id="toolu_2"),
        ]
        result = client.generate(messages)
        assert result.content == "merged"

        # 验证两个 tool_result 被合并到同一条 user 消息中
        call_args = mock_client.messages.create.call_args
        api_messages = call_args.kwargs["messages"]
        # 预期: [user, assistant(2 tool_use), user(2 tool_result)]
        assert len(api_messages) == 3
        assert api_messages[2]["role"] == "user"
        assert len(api_messages[2]["content"]) == 2
        assert api_messages[2]["content"][0]["type"] == "tool_result"
        assert api_messages[2]["content"][0]["tool_use_id"] == "toolu_1"
        assert api_messages[2]["content"][1]["type"] == "tool_result"
        assert api_messages[2]["content"][1]["tool_use_id"] == "toolu_2"


@dataclass
class MockTextDelta:
    type: str = "text_delta"
    text: str = ""


@dataclass
class MockInputJsonDelta:
    type: str = "input_json_delta"
    partial_json: str = ""
    index: int = 0


@dataclass
class MockContentBlockDelta:
    type: str = "content_block_delta"
    delta: object = None
    index: int = 0

    def __post_init__(self):
        if self.delta is None:
            self.delta = MockTextDelta()


@dataclass
class MockContentBlockStart:
    type: str = "content_block_start"
    index: int = 0
    content_block: object = None

    def __post_init__(self):
        if self.content_block is None:
            self.content_block = MockToolBlock()


@dataclass
class MockContentBlockStop:
    type: str = "content_block_stop"
    index: int = 0


@dataclass
class MockMessageDelta:
    type: str = "message_delta"
    _stop_reason: str = None
    delta: object = None

    def __post_init__(self):
        if self.delta is None:
            self.delta = MagicMock()
        self.delta.stop_reason = self._stop_reason


@dataclass
class MockMessageStart:
    type: str = "message_start"
    message: object = None


@dataclass
class MockMessageStop:
    type: str = "message_stop"


class TestAnthropicClientStream:
    """测试 Anthropic 客户端流式"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="anthropic")

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_stream_text_response(self, mock_anthropic_cls):
        """测试纯文本流式响应"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.return_value = [
            MockContentBlockDelta(delta=MockTextDelta(text="Hello")),
            MockContentBlockDelta(delta=MockTextDelta(text=" World")),
            MockMessageDelta(_stop_reason="end_turn"),
            MockMessageStop(),
        ]

        client = AnthropicClient()
        chunks = list(client.stream([Message(role="user", content="hi")]))

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].content == " World"
        assert chunks[2].is_final is True

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_stream_tool_call(self, mock_anthropic_cls):
        """测试工具调用流式响应"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.return_value = [
            MockContentBlockDelta(delta=MockTextDelta(text="我来读文件")),
            MockContentBlockStart(index=1, content_block=MockToolBlock(
                id="toolu_123", name="read_file", input={},
            )),
            MockContentBlockDelta(index=1, delta=MockInputJsonDelta(partial_json='{"path":')),
            MockContentBlockDelta(index=1, delta=MockInputJsonDelta(partial_json='"test.txt"}')),
            MockContentBlockStop(index=1),
            MockMessageDelta(_stop_reason="tool_use"),
            MockMessageStop(),
        ]

        client = AnthropicClient()
        chunks = list(client.stream(
            [Message(role="user", content="读 test.txt")],
            tools=[ToolDefinition(name="read_file", description="读文件", parameters={})],
        ))

        # 应该有文本块 + 工具调用块
        text_chunks = [c for c in chunks if c.content]
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(text_chunks) >= 1
        assert text_chunks[0].content == "我来读文件"

        # 最后一个工具调用 chunk 应包含完整 tool_call
        assert len(tool_chunks) >= 1
        last_tool_chunk = tool_chunks[-1]
        assert len(last_tool_chunk.tool_calls) == 1
        assert last_tool_chunk.tool_calls[0].function.name == "read_file"

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_stream_is_final(self, mock_anthropic_cls):
        """测试 is_final 标记"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.return_value = [
            MockContentBlockDelta(delta=MockTextDelta(text="done")),
            MockMessageDelta(_stop_reason="end_turn"),
            MockMessageStop(),
        ]

        client = AnthropicClient()
        chunks = list(client.stream([Message(role="user", content="hi")]))

        assert chunks[-1].is_final is True


@dataclass
class MockThinkingDelta:
    type: str = "thinking_delta"
    thinking: str = ""


@dataclass
class MockThinkingBlock:
    type: str = "thinking"
    thinking: str = ""


class TestAnthropicClientReasoning:
    """测试 Anthropic 客户端思维链 (thinking → reasoning)"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="anthropic")

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_normalize_out_thinking_to_reasoning(self, mock_anthropic_cls):
        """测试 thinking block 映射到 reasoning 字段"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_thinking_block = MockThinkingBlock(thinking="让我先思考...")
        mock_text_block = MockTextBlock(text="文件内容是 hello")
        mock_response = MagicMock(content=[mock_thinking_block, mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="读文件")])

        assert result.content == "文件内容是 hello"
        assert result.reasoning == "让我先思考..."

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_normalize_out_thinking_with_tool_use(self, mock_anthropic_cls):
        """测试 thinking + tool_use 时 reasoning 也提取"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_thinking_block = MockThinkingBlock(thinking="我需要读取文件...")
        mock_tool_block = MockToolBlock(id="toolu_1", name="read_file", input={"path": "test.txt"})
        mock_response = MagicMock(content=[mock_thinking_block, mock_tool_block], stop_reason="tool_use")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="读文件")])

        assert result.reasoning == "我需要读取文件..."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "read_file"

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_stream_thinking(self, mock_anthropic_cls):
        """测试流式 thinking delta 映射到 reasoning"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.return_value = [
            MockContentBlockDelta(index=0, delta=MockThinkingDelta(thinking="让我")),
            MockContentBlockDelta(index=0, delta=MockThinkingDelta(thinking="想想")),
            MockContentBlockDelta(delta=MockTextDelta(text="读取文件")),
            MockMessageDelta(_stop_reason="end_turn"),
            MockMessageStop(),
        ]

        client = AnthropicClient()
        chunks = list(client.stream([Message(role="user", content="读文件")]))

        # reasoning 增量返回（与 content 一致）
        reasoning_chunks = [c for c in chunks if c.reasoning]
        assert len(reasoning_chunks) == 2
        assert reasoning_chunks[0].reasoning == "让我"
        assert reasoning_chunks[1].reasoning == "想想"
        # 最后一个 chunk 是 is_final 信号
        assert chunks[-1].is_final is True
