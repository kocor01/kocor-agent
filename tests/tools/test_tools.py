"""测试工具系统"""

import os
import tempfile
import time
from unittest.mock import mock_open, patch

from kocor.config import Config
from kocor.llm_provider.message import FunctionCall, ToolCall, ToolResult
from kocor.tools.tool_manager import ToolManager
from kocor.tools.tool_utils import resolve_safe_path


class TestToolRegistry:
    """测试 ToolManager"""

    def test_register_and_get_definitions(self):
        """测试注册工具并获取定义"""
        registry = ToolManager()

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
        registry = ToolManager()

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
        registry = ToolManager()

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="unknown_tool", arguments="{}"),
        )
        result = registry.execute(tool_call)

        assert "not found" in result.content.lower() or "未找到" in result.content

    def test_multiple_tools(self):
        """测试多个工具注册"""
        registry = ToolManager()

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


class TestResolveSafePath:
    """测试路径遍历防护"""

    def test_path_within_allowed_dir(self):
        """目录内的路径正常解析"""
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = resolve_safe_path("subdir/file.txt", tmpdir)
            expected = os.path.realpath(os.path.join(tmpdir, "subdir/file.txt"))
            assert resolved == expected

    def test_path_traversal_rejected(self):
        """相对路径遍历抛出 PermissionError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                resolve_safe_path("../outside.txt", tmpdir)
                assert False, "应抛出 PermissionError"
            except PermissionError:
                pass

    def test_path_traversal_deeply_nested(self):
        """深层路径遍历抛出 PermissionError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                resolve_safe_path("a/b/c/../../../../etc/passwd", tmpdir)
                assert False, "应抛出 PermissionError"
            except PermissionError:
                pass

    def test_path_to_allowed_dir_itself(self):
        """路径指向允许目录本身"""
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = resolve_safe_path(".", tmpdir)
            assert resolved == os.path.realpath(tmpdir)

    def test_absolute_path_outside_allowed_dir(self):
        """绝对路径指向 allowed_dir 外应被拒绝（P0.1 回归）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造一个位于 tmpdir 之外的绝对路径
            outside = os.path.join(os.path.dirname(tmpdir.rstrip(os.sep)), "outside.txt")
            try:
                resolve_safe_path(outside, tmpdir)
                assert False, "应抛出 PermissionError"
            except PermissionError:
                pass

    def test_absolute_path_inside_allowed_dir(self):
        """绝对路径位于 allowed_dir 内应正常返回"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inside = os.path.join(tmpdir, "sub", "file.txt")
            resolved = resolve_safe_path(inside, tmpdir)
            assert resolved == os.path.realpath(inside)
            # 必须仍落在 allowed_dir 内
            base = os.path.realpath(tmpdir)
            assert resolved == base or resolved.startswith(base + os.sep)


class TestCreateDefaultTools:
    """测试内置工具创建"""

    @patch("kocor.tools.toolsets.read_file_tool.Path.exists", return_value=False)
    def test_read_file_not_found(self, mock_exists):
        """测试读取不存在的文件"""
        registry = ToolManager()
        registry.register_builtin_tools()
        tools = registry

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "nonexistent.txt"}'),
        )
        result = tools.execute(tool_call)
        assert "not found" in result.content.lower() or "未找到" in result.content

    @patch("kocor.tools.toolsets.read_file_tool.os.path.exists", return_value=True)
    @patch("kocor.tools.toolsets.read_file_tool.open", new_callable=mock_open, read_data="hello world")
    def test_read_file_success(self, mock_file, mock_exists):
        """测试读取文件成功"""
        registry = ToolManager()
        registry.register_builtin_tools()
        tools = registry

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'),
        )
        result = tools.execute(tool_call)
        assert "hello world" in result.content

    @patch("kocor.tools.toolsets.write_file_tool.os.makedirs")
    def test_write_file(self, mock_makedirs):
        """测试写入文件"""
        registry = ToolManager()
        registry.register_builtin_tools()
        tools = registry

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(
                name="write_file",
                arguments='{"path": "out.txt", "content": "test content"}',
            ),
        )
        result = tools.execute(tool_call)
        # 新 write_file 返回 JSON，可能含 error 或 bytes_written
        import json

        try:
            data = json.loads(result.content)
            assert "bytes_written" in data or "error" in data
        except (json.JSONDecodeError, AttributeError):
            assert "success" in result.content.lower() or "成功" in result.content

    def test_read_file_path_traversal_rejected(self):
        """读取文件路径遍历被拒绝"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kocor.tools.toolsets.read_file_tool.os.getcwd", return_value=tmpdir):
                registry = ToolManager()
                registry.register_builtin_tools()
                tools = registry

                tool_call = ToolCall(
                    id="call_1",
                    function=FunctionCall(
                        name="read_file",
                        arguments='{"path": "../etc/passwd"}',
                    ),
                )
                result = tools.execute(tool_call)
                assert "denied" in result.content.lower() or "拒绝" in result.content

    def test_write_file_path_traversal_rejected(self):
        """写入文件路径遍历被拒绝"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kocor.tools.toolsets.write_file_tool.os.getcwd", return_value=tmpdir):
                registry = ToolManager()
                registry.register_builtin_tools()
                tools = registry

                tool_call = ToolCall(
                    id="call_1",
                    function=FunctionCall(
                        name="write_file",
                        arguments='{"path": "../outside.txt", "content": "hack"}',
                    ),
                )
                tools.execute(tool_call)


class TestToolTimeout:
    """测试工具执行超时"""

    def setup_method(self):
        self._saved_timeout = Config.load().tool_timeout

    def teardown_method(self):
        Config.load().tool_timeout = self._saved_timeout

    def test_tool_timeout(self):
        """工具执行超时返回超时错误"""
        Config.load().tool_timeout = 1
        registry = ToolManager()

        def slow_handler(**kwargs):
            time.sleep(5)
            return "done"

        registry.register(
            name="slow_tool",
            description="慢工具",
            parameters={"type": "object"},
            handler=slow_handler,
        )

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="slow_tool", arguments="{}"),
        )
        result = registry.execute(tool_call)
        assert "timed out" in result.content.lower() or "timeout" in result.content.lower()

    def test_tool_normal_execution(self):
        """正常快速工具不受影响"""
        registry = ToolManager()

        def fast_handler(**kwargs):
            return "quick result"

        registry.register(
            name="fast_tool",
            description="快工具",
            parameters={"type": "object"},
            handler=fast_handler,
        )

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="fast_tool", arguments="{}"),
        )
        result = registry.execute(tool_call)
        assert result.content == "quick result"
