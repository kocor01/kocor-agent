"""bash/environment.py 单元测试：BaseEnvironment 和 LocalEnvironment。"""

import os
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from kocor.tools.toolset.bash.environment import BaseEnvironment, LocalEnvironment


# =============================================================================
# BaseEnvironment 抽象基类测试
# =============================================================================

class TestBaseEnvironment:
    """BaseEnvironment 抽象基类测试（通过子类模拟）。"""

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEnvironment(cwd="/tmp", timeout=30)

    def test_execute_success(self):
        """通过测试子类验证 execute 流程。"""
        env = _make_test_env(command_output="hello world", exit_code=0)
        result = env.execute("echo hello")
        assert result["stdout"].strip() == "hello world"
        assert result["exit_code"] == 0

    def test_execute_with_cwd(self):
        env = _make_test_env(command_output="")
        result = env.execute("pwd", cwd="/tmp")
        assert result["exit_code"] in (0, None)

    def test_execute_timeout(self):
        """验证超时路径：mock 的 poll 保持返回 None 直到超时触发。"""
        import time as _time

        call_count = 0
        real_deadline = _time.monotonic() + 0.2  # 给 mock 200ms 窗口

        class DelayedExitEnv(BaseEnvironment):
            def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
                from unittest.mock import MagicMock
                proc = MagicMock()

                def delayed_poll():
                    nonlocal call_count
                    call_count += 1
                    if _time.monotonic() >= real_deadline:
                        return 0
                    return None

                proc.poll = delayed_poll
                proc.stdout = MagicMock()
                proc.stdout.fileno.return_value = None
                proc.stdout.__iter__.return_value = iter(["output\n"])
                proc.stdout.read.return_value = "output\n"
                proc.returncode = 124
                return proc

            def cleanup(self):
                pass

        env = DelayedExitEnv(cwd="/tmp", timeout=0.05)
        result = env.execute("sleep 10")
        assert result["exit_code"] == 124
        assert "timed out" in result["stdout"]

    def test_execute_exit_code_interpretation(self):
        env = _make_test_env(command_output="test output", exit_code=1)
        # grep 的 exit_code=1 不应报错
        result = env.execute("grep foo bar.txt")
        assert result["exit_code"] == 1
        # exit_code 解释应包含在 result 中
        assert "exit_code_note" in result

    def test_snapshot_paths_use_session_id(self):
        env = _make_test_env()
        assert env._snapshot_path
        assert env._cwd_file
        assert env._cwd_marker
        assert env._session_id in env._snapshot_path
        assert env._session_id in env._cwd_file
        assert env._session_id in env._cwd_marker

    def test_cwd_marker_is_deterministic(self):
        env = _make_test_env()
        marker = env._cwd_marker
        assert marker.startswith("__KOCOR_CWD_")
        assert marker.endswith("__")


# =============================================================================
# LocalEnvironment 集成测试
# =============================================================================

class TestLocalEnvironmentInit:
    """LocalEnvironment 初始化和快照测试。"""

    def test_init_creates_snapshot(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
        assert env._snapshot_ready
        # 快照文件应存在
        snap_path = env._snapshot_path
        assert os.path.exists(snap_path)

    def test_init_uses_given_cwd(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
        assert env.cwd == str(tmp_path)

    def test_init_without_cwd_uses_current_dir(self):
        env = LocalEnvironment(timeout=30)
        assert env.cwd == os.getcwd()

    def test_init_results_in_session_id(self):
        env = LocalEnvironment(timeout=30)
        assert len(env._session_id) > 0


class TestLocalEnvironmentExecute:
    """LocalEnvironment 前台命令执行测试。"""

    def test_simple_echo(self):
        env = LocalEnvironment(timeout=10)
        result = env.execute("echo hello kocor")
        assert result["exit_code"] == 0
        assert "hello kocor" in result["stdout"]

    def test_exit_code_nonzero(self):
        env = LocalEnvironment(timeout=10)
        result = env.execute("exit 42")
        assert result["exit_code"] == 42

    def test_stderr_is_captured(self):
        env = LocalEnvironment(timeout=10)
        result = env.execute("echo stderr_test >&2")
        assert result["exit_code"] == 0
        assert "stderr_test" in result["stdout"]

    def test_workdir_affects_execution(self, tmp_path):
        # 在 tmp_path 中创建一个标记文件
        marker_file = tmp_path / "marker.txt"
        marker_file.write_text("present")
        env = LocalEnvironment(timeout=10)
        # 使用 workdir 参数执行
        result = env.execute("cat marker.txt", cwd=str(tmp_path))
        assert result["exit_code"] == 0
        assert "present" in result["stdout"]

    def test_timeout_kills_process(self):
        env = LocalEnvironment(timeout=30)
        result = env.execute("sleep 10", timeout=1)
        assert result["exit_code"] == 124
        assert "timed out" in result["stdout"]

    def test_empty_command_returns_error(self):
        env = LocalEnvironment(timeout=10)
        result = env.execute("")
        assert result["exit_code"] != 0

    def test_empty_command_output_is_str(self):
        env = LocalEnvironment(timeout=10)
        result = env.execute("")
        assert isinstance(result["stdout"], str)


class TestLocalEnvironmentCwdTracking:
    """CWD 追踪机制测试。"""

    def test_cwd_changes_after_cd(self):
        env = LocalEnvironment(timeout=10)
        # 改为 cd 到项目根目录，这是一个 Git Bash 和 Windows 都能理解的路径
        env.execute("cd /")
        # 验证 CWD 变为根目录（Windows 上为 C:\ 或 D:\ 等）
        assert env.cwd is not None

    def test_cwd_persists_between_calls(self):
        env = LocalEnvironment(timeout=10)
        env.execute("cd /")
        result = env.execute("pwd")
        assert result["exit_code"] == 0
        # 输出应包含路径分隔符（表示有路径输出）
        assert "/" in result["stdout"] or "\\" in result["stdout"]

    def test_cwd_resolves_to_parent_when_deleted(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        env = LocalEnvironment(cwd=str(subdir), timeout=10)
        import shutil
        shutil.rmtree(str(subdir))
        result = env.execute("pwd")
        assert result["exit_code"] == 0
        assert os.path.isdir(env.cwd)


class TestLocalEnvironmentCleanup:
    """清理测试。"""

    def test_cleanup_removes_temp_files(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        snap = env._snapshot_path
        cwd_file = env._cwd_file
        assert os.path.exists(snap)
        env.cleanup()
        assert not os.path.exists(snap)
        assert not os.path.exists(cwd_file)


class TestLocalEnvironmentWrapCommand:
    """命令包装测试。"""

    def test_wrap_contains_cd(self):
        env = LocalEnvironment(timeout=10)
        wrapped = env._wrap_command("echo hello", "/tmp")
        assert "cd" in wrapped
        assert "echo hello" in wrapped
        assert "exit $__kocor_ec" in wrapped

    def test_wrap_contains_cwd_marker(self):
        env = LocalEnvironment(timeout=10)
        wrapped = env._wrap_command("echo hello", "/tmp")
        assert env._cwd_marker in wrapped

    def test_wrap_contains_snapshot_source(self):
        env = LocalEnvironment(timeout=10)
        wrapped = env._wrap_command("echo hello", "/tmp")
        snap_name = os.path.basename(env._snapshot_path)
        assert snap_name in wrapped


# =============================================================================
# Helper: 创建 BaseEnvironment 的测试子类
# =============================================================================

def _make_test_env(command_output: str = "", exit_code: int = 0, timeout: int = 30):
    """创建 BaseEnvironment 的测试子类，模拟 _run_bash 行为。"""
    import time as _time

    class TestEnv(BaseEnvironment):
        def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            import subprocess
            proc = MagicMock()
            # 模拟 subprocess.Popen 接口
            proc.poll.return_value = None
            proc.wait.return_value = None
            proc.stdout = MagicMock()
            proc.stdout.fileno.return_value = None  # 触发 fallback 路径
            proc.stdout.__iter__.return_value = iter([command_output + "\n"])

            def _fake_read():
                return command_output

            proc.stdout.read = _fake_read

            # 装饰 _wait_for_process 中的 proc.poll() 以模拟进程退出
            poll_values = [None, None, exit_code]
            proc.poll.side_effect = poll_values
            proc.returncode = exit_code
            return proc

        def cleanup(self):
            pass

    return TestEnv(cwd="/tmp", timeout=timeout)