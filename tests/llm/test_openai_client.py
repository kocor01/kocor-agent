"""测试 OpenAI 客户端"""

from unittest.mock import MagicMock, patch

from kocor.config import Config
from kocor.llm_provider.tool_definition import ToolDefinition
from kocor.llm_provider.openai_client import OpenAIClient
from kocor.llm_provider.message import FunctionCall, Message, ToolCall


class TestOpenAIClient:
    """测试 OpenAI 客户端"""

    def _make_config(self, **kwargs) -> Config:
        defaults = {"provider": "openai"}
        defaults.update(kwargs)
        return Config(**defaults)

    def test_provider(self):
        client = OpenAIClient(self._make_config())
        assert client.provider == "openai"

    
    @patch("kocor.llm_provider.openai_client.OpenAI")
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

    @patch("kocor.llm_provider.openai_client.OpenAI")
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

    @patch("kocor.llm_provider.openai_client.OpenAI")
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

    @patch("kocor.llm_provider.openai_client.OpenAI")
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


class MockOpenAIChunk:
    """Mock OpenAI streaming chunk"""

    def __init__(self, content=None, tool_calls=None, finish_reason=None, reasoning=None, reasoning_content=None):
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls
        delta.index = 0
        delta.reasoning = reasoning
        delta.reasoning_content = reasoning_content
        self.choices = [MagicMock(delta=delta, finish_reason=finish_reason)]


class TestOpenAIClientStream:
    """测试 OpenAI 客户端流式"""

    def _make_config(self, **kwargs) -> Config:
        defaults = {"provider": "openai"}
        defaults.update(kwargs)
        return Config(**defaults)

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_stream_text_response(self, mock_openai_cls):
        """测试纯文本流式响应"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # 模拟流式输出: "Hello" + " World" + end
        mock_client.chat.completions.create.return_value = [
            MockOpenAIChunk(content="Hello"),
            MockOpenAIChunk(content=" World"),
            MockOpenAIChunk(finish_reason="stop"),
        ]

        client = OpenAIClient(self._make_config())
        chunks = list(client.stream([Message(role="user", content="hi")]))

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].content == " World"
        assert chunks[2].is_final is True

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_stream_tool_call(self, mock_openai_cls):
        """测试工具调用流式响应"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # 模拟工具调用流式输出
        mock_tc_delta = MagicMock()
        mock_tc_delta.id = "call_123"
        mock_tc_delta.type = "function"
        mock_tc_delta.function.name = "read_file"
        mock_tc_delta.function.arguments = '{"path": "test.txt"}'

        mock_client.chat.completions.create.return_value = [
            MockOpenAIChunk(content="我来读文件"),
            MockOpenAIChunk(
                tool_calls=[mock_tc_delta],
                finish_reason="tool_calls",
            ),
        ]

        client = OpenAIClient(self._make_config())
        chunks = list(client.stream(
            [Message(role="user", content="读 test.txt")],
            tools=[ToolDefinition(name="read_file", description="读文件", parameters={})],
        ))

        assert len(chunks) == 2
        assert chunks[0].content == "我来读文件"
        assert chunks[1].is_final is True
        assert len(chunks[1].tool_calls) == 1
        assert chunks[1].tool_calls[0].function.name == "read_file"

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_stream_empty_content(self, mock_openai_cls):
        """测试空内容 chunk 仍被 yield（携带 is_final 信号）"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.return_value = [
            MockOpenAIChunk(content=None),
            MockOpenAIChunk(content="hello"),
            MockOpenAIChunk(finish_reason="stop"),
        ]

        client = OpenAIClient(self._make_config())
        chunks = list(client.stream([Message(role="user", content="hi")]))

        # 空 content 的 chunk 仍 yield，因为 is_final 信号在最后一个 chunk 上
        assert len(chunks) == 3
        assert chunks[0].content == ""
        assert chunks[1].content == "hello"
        assert chunks[2].is_final is True

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_normalize_in_system_message(self, mock_openai_cls):
        """测试 _normalize_in 处理 system 消息"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "hello"
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="hi"),
        ]
        client.generate(messages)

        call_args = mock_client.chat.completions.create.call_args
        msgs = call_args.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "你是助手"

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_normalize_in_tool_result(self, mock_openai_cls):
        """测试 _normalize_in 处理 tool 消息"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "done"
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{}')),
            ]),
            Message(role="tool", content="result", tool_call_id="call_1"),
        ]
        client.generate(messages)

        call_args = mock_client.chat.completions.create.call_args
        msgs = call_args.kwargs["messages"]
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["content"] == "result"
        assert msgs[2]["tool_call_id"] == "call_1"

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_normalize_in_assistant_reasoning(self, mock_openai_cls):
        """测试 _normalize_in 传递 assistant reasoning"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "done"
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="结果", reasoning="让我想想..."),
        ]
        client.generate(messages)

        call_args = mock_client.chat.completions.create.call_args
        msgs = call_args.kwargs["messages"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["reasoning"] == "让我想想..."

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_normalize_out_reasoning(self, mock_openai_cls):
        """测试 _normalize_out 提取 reasoning"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = "文件内容是 hello"
        mock_message.reasoning = "让我先读取文件..."
        mock_message.tool_calls = None
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(self._make_config())
        result = client.generate([Message(role="user", content="读文件")])

        assert result.content == "文件内容是 hello"
        assert result.reasoning == "让我先读取文件..."

    @patch("kocor.llm_provider.openai_client.OpenAI")
    def test_stream_reasoning(self, mock_openai_cls):
        """测试流式输出提取 reasoning"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_tc_delta = MagicMock()
        mock_tc_delta.index = 0

        mock_client.chat.completions.create.return_value = [
            MockOpenAIChunk(content="让我"),
            MockOpenAIChunk(reasoning="先思考一下"),
            MockOpenAIChunk(content="读取文件"),
            MockOpenAIChunk(tool_calls=[mock_tc_delta], finish_reason="tool_calls"),
        ]

        client = OpenAIClient(self._make_config())
        chunks = list(client.stream([Message(role="user", content="读文件")]))

        # reasoning 增量返回（与 content 一致）
        reasoning_chunks = [c for c in chunks if c.reasoning]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0].reasoning == "先思考一下"
