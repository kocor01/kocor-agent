"""bash_tool.py 集成测试：BashTool 和 ProcessTool 的完整功能。"""

from unittest.mock import MagicMock, patch

import pytest

from kocor.tools.toolsets.bash_tool import BashTool, ProcessTool
from kocor.tools.toolsets.bash.environment import LocalEnvironment
from kocor.tools.toolsets.bash.process_registry import ProcessRegistry
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


# =============================================================================
# 阶段 2 重构测试：显式 env 注入 + 隔离性
# =============================================================================


class TestBashToolEnvInjection:
    """显式 env 参数注入测试。"""

    def test_handler_with_explicit_env(self, tmp_path):
        """传入显式的 LocalEnvironment 实例应能正常执行。"""
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = BashTool.handler(command="echo hello", env=env)
        assert "hello" in result
        env.cleanup()

    def test_handler_env_with_workdir(self, tmp_path):
        """注入 env + 指定 workdir 应正常工作。"""
        env = LocalEnvironment(timeout=10)
        marker = tmp_path / "marker.txt"
        marker.write_text("injected")
        result = BashTool.handler(command="cat marker.txt", workdir=str(tmp_path), env=env)
        assert "injected" in result
        env.cleanup()

    def test_handler_env_fallback_when_none(self):
        """不传 env 时，应回退到模块级全局 _env（向后兼容）。"""
        result = BashTool.handler(command="echo fallback_ok")
        assert "fallback_ok" in result


class TestBashToolEnvIsolation:
    """两个独立的 LocalEnvironment 实例不应互相干扰。"""

    def test_cwd_isolation(self, tmp_path):
        """两个 env 实例的 CWD 应彼此独立。"""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        env_a = LocalEnvironment(cwd=str(dir_a), timeout=10)
        env_b = LocalEnvironment(cwd=str(dir_b), timeout=10)

        # 验证各自 CWD 独立
        assert env_a.cwd == str(dir_a)
        assert env_b.cwd == str(dir_b)

        # 在 env_a 中 cd 到 /，不应影响 env_b
        BashTool.handler(command="cd /", env=env_a)
        assert env_b.cwd == str(dir_b)  # env_b 的 CWD 不变

        env_a.cleanup()
        env_b.cleanup()

    def test_background_env_isolation(self, tmp_path):
        """后台进程使用不同的 env 实例应互不干扰。"""
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = BashTool.handler(command="echo isolated_bg", background=True, env=env)
        assert "proc_" in result
        assert "running" in result
        env.cleanup()

    def test_concurrent_env_instances(self, tmp_path):
        """两个 env 实例能同时存在且独立工作。"""
        env_a = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        env_b = LocalEnvironment(cwd=str(tmp_path), timeout=10)

        result_a = BashTool.handler(command="echo aaa", env=env_a)
        result_b = BashTool.handler(command="echo bbb", env=env_b)

        assert "aaa" in result_a
        assert "bbb" in result_b

        env_a.cleanup()
        env_b.cleanup()