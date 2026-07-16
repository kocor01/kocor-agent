"""BundledTool handler_factory 注册与依赖注入测试。"""

from __future__ import annotations

from kocor.tools.tool_manager import ToolManager


class TestBundledToolRegistration:
    def test_register_all_bundled_includes_all_tools(self):
        """register_builtin_tools 应注册所有核心工具。"""
        tm = ToolManager()
        tm.register_builtin_tools()
        definitions = tm.get_definitions()
        names = {d.name for d in definitions}
        assert "read_file" in names
        assert "write_file" in names
        assert "patch_file" in names
        assert "bash" in names
        assert "search_files" in names
        assert "process" in names
        assert "memory" in names
        assert "todo" in names

    def test_handler_factory_injects_deps(self):
        """handler_factory 应正确注入依赖。"""
        tm = ToolManager()
        tm.register_builtin_tools()

        for name, handler in tm._handlers.items():
            if name == "read_file":
                result = handler(path="/nonexistent")
                assert isinstance(result, str)
            elif name == "write_file":
                result = handler(path="/nonexistent", content="test")
                assert isinstance(result, str)

    def test_handler_execution_returns_string(self):
        """所有已注册的 handler 应返回字符串。

        跳过 bash/process（触发子进程，Windows GBK 编码兼容问题），
        也跳过 cronjob（需创建 cron worker 子进程）。
        """
        tm = ToolManager()
        tm.register_builtin_tools()

        skipped = {"bash", "process", "cronjob"}
        for name, handler in tm._handlers.items():
            if name in skipped:
                continue
            if name == "search_files":
                result = handler(pattern="test")
            elif name == "read_file":
                result = handler(path="/nonexistent")
            elif name == "write_file":
                result = handler(path="/nonexistent", content="test")
            elif name == "patch_file":
                result = handler(path="/nonexistent", old_string="a", new_string="b")
            elif name == "todo":
                result = handler(todos=[])
            elif name == "memory":
                result = handler(operations=[])
            else:
                continue
            assert isinstance(result, str), f"{name} handler should return str, got {type(result)}"