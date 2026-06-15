"""测试消息数据模型"""

from kocor.message import Message, ToolCall, FunctionCall, ToolResult


class TestMessage:
    """测试 Message 数据类"""

    def test_create_user_message(self):
        msg = Message(role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"

    def test_create_system_message(self):
        msg = Message(role="system", content="你是助手")
        assert msg.role == "system"
        assert msg.content == "你是助手"

    def test_create_assistant_message_with_content(self):
        msg = Message(role="assistant", content="你好，我是助手")
        assert msg.role == "assistant"
        assert msg.content == "你好，我是助手"

    def test_create_assistant_message_without_content(self):
        msg = Message(role="assistant", content="")
        assert msg.role == "assistant"
        assert msg.content == ""

    def test_create_tool_message(self):
        msg = Message(role="tool", content="结果", tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.content == "结果"
        assert msg.tool_call_id == "call_123"

    def test_create_assistant_message_with_reasoning(self):
        """测试 assistant 消息带思维链"""
        msg = Message(role="assistant", content="文件内容是 hello", reasoning="让我先读取文件...")
        assert msg.role == "assistant"
        assert msg.content == "文件内容是 hello"
        assert msg.reasoning == "让我先读取文件..."

    def test_reasoning_default_empty(self):
        """测试 reasoning 默认为空字符串"""
        msg = Message(role="user", content="hi")
        assert msg.reasoning == ""


class TestToolCall:
    """测试 ToolCall 数据类"""

    def test_create_tool_call(self):
        call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'),
        )
        assert call.id == "call_1"
        assert call.type == "function"
        assert call.function.name == "read_file"
        assert call.function.arguments == '{"path": "test.txt"}'

    def test_tool_call_default_type(self):
        call = ToolCall(
            id="call_1",
            function=FunctionCall(name="write_file", arguments='{"path": "a.txt"}'),
        )
        assert call.type == "function"


class TestFunctionCall:
    """测试 FunctionCall 数据类"""

    def test_create_function_call(self):
        fc = FunctionCall(name="run_python", arguments='{"code": "print(1)"}')
        assert fc.name == "run_python"
        assert fc.arguments == '{"code": "print(1)"}'


class TestToolResult:
    """测试 ToolResult 数据类"""

    def test_create_tool_result(self):
        result = ToolResult(tool_call_id="call_1", content="文件内容: hello")
        assert result.tool_call_id == "call_1"
        assert result.content == "文件内容: hello"


class TestStreamChunk:
    """测试 StreamChunk 数据类"""

    def test_stream_chunk_default_values(self):
        """测试默认值: content="" tool_calls=[] is_final=False"""
        from kocor.message import StreamChunk

        chunk = StreamChunk()
        assert chunk.content == ""
        assert chunk.tool_calls == []
        assert chunk.is_final is False

    def test_stream_chunk_with_content(self):
        """测试带增量文本的 chunk"""
        from kocor.message import StreamChunk

        chunk = StreamChunk(content="你好")
        assert chunk.content == "你好"
        assert chunk.is_final is False

    def test_stream_chunk_is_final(self):
        """测试 is_final 标记"""
        from kocor.message import StreamChunk

        chunk = StreamChunk(content="结束", is_final=True)
        assert chunk.is_final is True

    def test_stream_chunk_with_tool_calls(self):
        """测试带工具调用的 chunk"""
        from kocor.message import StreamChunk, ToolCall, FunctionCall

        chunk = StreamChunk(
            tool_calls=[
                ToolCall(
                    id="toolu_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )
            ]
        )
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].function.name == "read_file"
