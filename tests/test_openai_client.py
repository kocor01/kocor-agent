"""测试 OpenAI 客户端"""

from unittest.mock import MagicMock, patch

from kocor.config import LLMConfig
from kocor.llm_client import ToolDefinition
from kocor.message import Message, ToolCall, ToolResult, FunctionCall
from kocor.openai_client import OpenAIClient


class TestOpenAIClient:
    """测试 OpenAI 客户端"""

    def _make_config(self, **kwargs) -> LLMConfig:
        defaults = {"provider": "openai"}
        defaults.update(kwargs)
        return LLMConfig(**defaults)

    def test_provider(self):
        client = OpenAIClient(self._make_config())
        assert client.provider == "openai"

    @patch("kocor.openai_client.OpenAI")
    def test_generate_text_response(self, mock_openai_cls):
        """测试纯文本响应"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock OpenAI API 响应
        mock_message = MagicMock()
        mock_message.content = "你好，我是助手"
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        result = client.generate([Message(role="user", content="你好")])

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "你好，我是助手"
        assert result.tool_calls == []

    @patch("kocor.openai_client.OpenAI")
    def test_generate_tool_call_response(self, mock_openai_cls):
        """测试工具调用响应"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock OpenAI API 工具调用响应
        mock_function = MagicMock()
        mock_function.name = "read_file"
        mock_function.arguments = '{"path": "test.txt"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        result = client.generate([Message(role="user", content="读 test.txt")])

        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].function.name == "read_file"
        assert result.tool_calls[0].function.arguments == '{"path": "test.txt"}'

    @patch("kocor.openai_client.OpenAI")
    def test_generate_with_tools(self, mock_openai_cls):
        """测试传入工具定义"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "我来读文件"
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        tools = [
            ToolDefinition(name="read_file", description="读文件", parameters={"type": "object", "properties": {}})
        ]
        client.generate(
            [Message(role="user", content="读文件")], tools=tools
        )

        # 验证调用了 create 并传入了 tools 参数
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs
        tools_arg = call_args.kwargs["tools"]
        assert len(tools_arg) == 1
        assert tools_arg[0]["type"] == "function"
        assert tools_arg[0]["function"]["name"] == "read_file"

    @patch("kocor.openai_client.OpenAI")
    def test_generate_with_temperature(self, mock_openai_cls):
        """测试 temperature 参数传递"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "hello"
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        client.generate(
            [Message(role="user", content="hi")],
            temperature=0.7,
            max_tokens=1024,
        )

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 1024
