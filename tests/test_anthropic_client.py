"""测试 Anthropic 客户端"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from kocor.config import LLMConfig
from kocor.llm_client import ToolDefinition
from kocor.message import Message, ToolCall, FunctionCall
from kocor.anthropic_client import AnthropicClient


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

    def _make_config(self, **kwargs) -> LLMConfig:
        defaults = {"provider": "anthropic"}
        defaults.update(kwargs)
        return LLMConfig(**defaults)

    def test_provider(self):
        client = AnthropicClient(self._make_config())
        assert client.provider == "anthropic"

    @patch("kocor.anthropic_client.Anthropic")
    def test_generate_text_response(self, mock_anthropic_cls):
        """测试纯文本响应"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="你好，我是助手")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(self._make_config())
        result = client.generate([Message(role="user", content="你好")])

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "你好，我是助手"
        assert result.tool_calls == []

    @patch("kocor.anthropic_client.Anthropic")
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

        client = AnthropicClient(self._make_config())
        result = client.generate([Message(role="user", content="读 test.txt")])

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].function.name == "read_file"
        import json
        assert json.loads(result.tool_calls[0].function.arguments) == {"path": "test.txt"}

    @patch("kocor.anthropic_client.Anthropic")
    def test_generate_with_tools(self, mock_anthropic_cls):
        """测试传入工具定义"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="我来读文件")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(self._make_config())
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

    @patch("kocor.anthropic_client.Anthropic")
    def test_generate_system_message(self, mock_anthropic_cls):
        """测试 system 消息传递"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_text_block = MockTextBlock(text="hello")
        mock_response = MagicMock(content=[mock_text_block], stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(self._make_config())
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="hi"),
        ]
        client.generate(messages)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["system"] == "你是助手"

    @patch("kocor.anthropic_client.Anthropic")
    def test_normalize_tool_result(self, mock_anthropic_cls):
        """测试工具结果格式归一化"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MockTextBlock(text="done")],
            stop_reason="end_turn",
        )

        client = AnthropicClient(self._make_config())
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="toolu_1", function=FunctionCall(name="read_file", arguments='{}')),
            ]),
            Message(role="tool", content="file content", tool_call_id="toolu_1"),
        ]
        result = client.generate(messages)
        assert result.content == "done"
