"""测试工具系统"""

import json
from unittest.mock import patch, mock_open

from kocor.message import FunctionCall, ToolCall, ToolResult
from kocor.tools import ToolRegistry, create_default_tools


class TestToolRegistry:
    """测试 ToolRegistry"""

    def test_register_and_get_definitions(self):
        """测试注册工具并获取定义"""
        registry = ToolRegistry()

        def handler(**kwargs):
            return "result"

        registry.register(
            name="test_tool",
            description="测试工具",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=handler,
        )

        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].name == "test_tool"
        assert defs[0].description == "测试工具"

    def test_execute_registered_tool(self):
        """测试执行已注册工具"""
        registry = ToolRegistry()

        def add_numbers(**kwargs):
            a = kwargs.get("a", 0)
            b = kwargs.get("b", 0)
            return str(a + b)

        registry.register(
            name="add",
            description="加法",
            parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
            handler=add_numbers,
        )

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="add", arguments='{"a": 3, "b": 4}'),
        )
        result = registry.execute(tool_call)

        assert isinstance(result, ToolResult)
        assert result.tool_call_id == "call_1"
        assert result.content == "7"

    def test_execute_unknown_tool(self):
        """测试执行未注册工具"""
        registry = ToolRegistry()

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="unknown_tool", arguments='{}'),
        )
        result = registry.execute(tool_call)

        assert "not found" in result.content.lower() or "未找到" in result.content

    def test_multiple_tools(self):
        """测试多个工具注册"""
        registry = ToolRegistry()

        registry.register(
            name="tool_a",
            description="工具A",
            parameters={"type": "object"},
            handler=lambda **kwargs: "a",
        )
        registry.register(
            name="tool_b",
            description="工具B",
            parameters={"type": "object"},
            handler=lambda **kwargs: "b",
        )

        defs = registry.get_definitions()
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"tool_a", "tool_b"}


class TestCreateDefaultTools:
    """测试内置工具创建"""

    @patch("kocor.tools.os.path.exists")
    def test_read_file_not_found(self, mock_exists):
        """测试读取不存在的文件"""
        mock_exists.return_value = False
        tools = create_default_tools()

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "nonexistent.txt"}'),
        )
        result = tools.execute(tool_call)
        assert "not found" in result.content.lower() or "未找到" in result.content

    @patch("kocor.tools.open", new_callable=mock_open, read_data="hello world")
    @patch("kocor.tools.os.path.exists")
    def test_read_file_success(self, mock_exists, mock_file):
        """测试读取文件成功"""
        mock_exists.return_value = True
        tools = create_default_tools()

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'),
        )
        result = tools.execute(tool_call)
        assert "hello world" in result.content

    @patch("kocor.tools.os.path.exists")
    def test_write_file(self, mock_exists):
        """测试写入文件"""
        import os

        mock_exists.return_value = True
        tools = create_default_tools()

        try:
            tool_call = ToolCall(
                id="call_1",
                function=FunctionCall(name="write_file", arguments='{"path": "out.txt", "content": "test content"}'),
            )
            result = tools.execute(tool_call)
            assert "success" in result.content.lower() or "成功" in result.content
        finally:
            if os.path.exists("out.txt"):
                os.remove("out.txt")

    @patch("kocor.tools.subprocess.run")
    def test_run_python_success(self, mock_run):
        """测试执行 Python 代码成功"""
        mock_run.return_value = type("MockResult", (), {
            "returncode": 0,
            "stdout": "42\n",
            "stderr": "",
        })()

        tools = create_default_tools()
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="run_python", arguments='{"code": "print(42)"}'),
        )
        result = tools.execute(tool_call)
        assert "42" in result.content

    @patch("kocor.tools.subprocess.run")
    def test_run_python_failure(self, mock_run):
        """测试执行 Python 代码失败"""
        mock_run.return_value = type("MockResult", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": "NameError: name 'x' is not defined",
        })()

        tools = create_default_tools()
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="run_python", arguments='{"code": "print(x)"}'),
        )
        result = tools.execute(tool_call)
        assert "NameError" in result.content
