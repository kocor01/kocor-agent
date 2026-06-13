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
