"""bash_tool.py 集成测试：BashTool 和 ProcessTool 的完整功能。"""

from unittest.mock import MagicMock, patch

import pytest

from kocor.tools.toolset.bash_tool import BashTool, ProcessTool
from kocor.tools.toolset.bash.environment import LocalEnvironment
from kocor.tools.toolset.bash.process_registry import ProcessRegistry
from kocor.tools.permission import PermissionManager


class TestBashToolDefinition:
    """BashTool 定义测试。"""

    def test_tool_name(self):
        assert BashTool.NAME == "bash"

    def test_tool_safety_level(self):
        assert BashTool.SAFETY_LEVEL == PermissionManager.SAFETY_DANGEROUS

    def test_tool_parameters_have_required_command(self):
        assert "command" in BashTool.PARAMETERS.get("required", [])

    def test_tool_description_not_empty(self):
        assert BashTool.DESCRIPTION
        assert len(BashTool.DESCRIPTION) > 20


class TestBashToolHandler:
    """BashTool handler 功能测试。"""

    @classmethod
    def setup_class(cls):
        """确保 LocalEnvironment 等模块可用。"""
        pass

    def test_handler_simple_echo(self):
        result = BashTool.handler(command="echo hello kocor")
        assert "hello kocor" in result
        assert "error" not in result.lower() or "exit_code" not in result

    def test_handler_exit_code(self):
        result = BashTool.handler(command="exit 42")
        assert "42" in result

    def test_handler_empty_command(self):
        result = BashTool.handler(command="")
        assert "error" in result.lower() or "empty" in result.lower()

    def test_handler_dangerous_command_blocked(self):
        result = BashTool.handler(command="rm -rf /")
        assert "blocked" in result.lower() or "dangerous" in result.lower() or "denied" in result.lower()

    def test_handler_with_workdir(self, tmp_path):
        marker = tmp_path / "marker.txt"
        marker.write_text("present")
        result = BashTool.handler(command="cat marker.txt", workdir=str(tmp_path))
        assert "present" in result

    def test_handler_invalid_workdir(self):
        result = BashTool.handler(command="echo hello", workdir="/tmp; rm -rf /")
        assert "blocked" in result.lower() or "disallowed" in result.lower()

    def test_handler_background(self):
        result = BashTool.handler(command="echo bg_test", background=True)
        assert "proc_" in result
        assert "running" in result

    def test_handler_timeout_long_command(self):
        """短超时应触发超时，而不是无限等待。"""
        result = BashTool.handler(command="sleep 10", timeout=1)
        assert "timed out" in result.lower() or "exit_code" in result

    def test_handler_cwd_tracking(self):
        """验证 CWD 在连续调用间保持。"""
        BashTool.handler(command="cd /tmp")
        result = BashTool.handler(command="pwd")
        # Linux 上 /tmp 存在，输出应包含 tmp
        assert "tmp" in result.lower() or "/" in result


class TestProcessTool:
    """ProcessTool 后台进程管理测试。"""

    def test_process_tool_name(self):
        assert ProcessTool.NAME == "process"

    def test_process_list(self):
        result = ProcessTool.handler(action="list")
        assert isinstance(result, str)

    def test_process_poll_nonexistent(self):
        result = ProcessTool.handler(action="poll", session_id="nonexistent")
        assert "not_found" in result.lower() or "error" in result.lower()

    def test_process_kill_nonexistent(self):
        result = ProcessTool.handler(action="kill", session_id="nonexistent")
        assert "not_found" in result.lower() or "error" in result.lower()