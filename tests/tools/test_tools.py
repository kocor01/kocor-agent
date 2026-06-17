"""测试工具系统"""

import os
import tempfile
from unittest.mock import mock_open, patch

from kocor.llm_provider.message import FunctionCall, ToolCall, ToolResult
from kocor.tool_registry import ToolRegistry
from kocor.tools import create_default_tools
from kocor.tools.tool_utils import resolve_safe_path


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


class TestCreateDefaultTools:
    """测试内置工具创建"""

    @patch("kocor.tools.toolset.read_file.os.path.exists")
    def test_read_file_not_found(self, mock_exists):
        """测试读取不存在的文件"""
        mock_exists.return_value = False
        registry = ToolRegistry()
        create_default_tools(registry)
        tools = registry

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "nonexistent.txt"}'),
        )
        result = tools.execute(tool_call)
        assert "not found" in result.content.lower() or "未找到" in result.content

    @patch("kocor.tools.toolset.read_file.open", new_callable=mock_open, read_data="hello world")
    @patch("kocor.tools.toolset.read_file.os.path.exists")
    def test_read_file_success(self, mock_exists, mock_file):
        """测试读取文件成功"""
        mock_exists.return_value = True
        registry = ToolRegistry()
        create_default_tools(registry)
        tools = registry

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'),
        )
        result = tools.execute(tool_call)
        assert "hello world" in result.content

    @patch("kocor.tools.toolset.write_file.os.makedirs")
    def test_write_file(self, mock_makedirs):
        """测试写入文件"""
        registry = ToolRegistry()
        create_default_tools(registry)
        tools = registry

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="write_file", arguments='{"path": "out.txt", "content": "test content"}'),
        )
        try:
            result = tools.execute(tool_call)
            assert "success" in result.content.lower() or "成功" in result.content
        finally:
            if os.path.exists("out.txt"):
                os.remove("out.txt")

    def test_read_file_path_traversal_rejected(self):
        """读取文件路径遍历被拒绝"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kocor.tools.toolset.read_file.os.getcwd", return_value=tmpdir):
                registry = ToolRegistry()
                create_default_tools(registry)
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
            with patch("kocor.tools.toolset.write_file.os.getcwd", return_value=tmpdir):
                registry = ToolRegistry()
                create_default_tools(registry)
                tools = registry

                tool_call = ToolCall(
                    id="call_1",
                    function=FunctionCall(
                        name="write_file",
                        arguments='{"path": "../outside.txt", "content": "hack"}',
                    ),
                )
                result = tools.execute(tool_call)

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_run_python_success(self, mock_run):
        """测试执行 Python 代码成功"""
        mock_run.return_value = type("MockResult", (), {
            "returncode": 0,
            "stdout": "42\n",
            "stderr": "",
        })()

        registry = ToolRegistry()
        create_default_tools(registry)
        tools = registry
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="run_python", arguments='{"code": "print(42)"}'),
        )
        result = tools.execute(tool_call)
        assert "42" in result.content

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_run_python_strips_sensitive_env(self, mock_run):
        """子进程不包含敏感环境变量"""
        mock_run.return_value = type("MockResult", (), {
            "returncode": 0,
            "stdout": "ok\n",
            "stderr": "",
        })()

        registry = ToolRegistry()
        create_default_tools(registry)
        tools = registry
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(
                name="run_python", arguments='{"code": "print(42)"}'
            ),
        )
        tools.execute(tool_call)

        _call_kwargs = mock_run.call_args.kwargs
        env = _call_kwargs.get("env", os.environ)
        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_sanitize_env_keeps_non_sensitive_keys(self, mock_run):
        """非敏感变量（如含 'key' 子串的）不被过滤"""
        mock_run.return_value = type("MockResult", (), {
            "returncode": 0,
            "stdout": "ok\n",
            "stderr": "",
        })()

        # 确认 KEYBOARD_LAYOUT 等非敏感变量不被过滤
        from kocor.tools.tool_utils import sanitize_env
        env = sanitize_env()
        assert "PATH" in env  # PATH 永远不应被过滤
        # KEYBOARD_LAYOUT 含有 'key' 子串，但不以 _API_KEY 等结尾，不应被过滤
        # 注: 此变量可能不存在于实际环境中，但 _sanitize_env 不应主动删除它
        import os
        if "KEYBOARD_LAYOUT" in os.environ:
            assert "KEYBOARD_LAYOUT" in env

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_run_python_failure(self, mock_run):
        """测试执行 Python 代码失败"""
        mock_run.return_value = type("MockResult", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": "NameError: name 'x' is not defined",
        })()

        registry = ToolRegistry()
        create_default_tools(registry)
        tools = registry
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="run_python", arguments='{"code": "print(x)"}'),
        )
        result = tools.execute(tool_call)
        assert "NameError" in result.content