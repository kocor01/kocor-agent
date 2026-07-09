"""bash/process_registry.py 单元测试。"""

import time
from unittest.mock import MagicMock, patch

import pytest

from kocor.tools.toolsets.bash.process_registry import (
    ProcessRegistry,
    ProcessSession,
    MAX_OUTPUT_CHARS,
)


# env.execute() 返回 {"stdout": ..., "exit_code": ...}
_MOCK_ENV_RESULT = {"stdout": "12345\n", "exit_code": 0}


class TestProcessSession:
    """ProcessSession 数据结构测试。"""

    def test_default_fields(self):
        session = ProcessSession(id="proc_test", command="echo hello")
        assert session.id == "proc_test"
        assert session.command == "echo hello"
        assert session.exited is False
        assert session.exit_code is None
        assert session.output_buffer == ""

    def test_max_output_chars_default(self):
        session = ProcessSession(id="proc_test", command="echo hello")
        assert session.max_output_chars == MAX_OUTPUT_CHARS


class TestProcessRegistry:
    """ProcessRegistry 核心功能测试。"""

    def setup_method(self):
        self.registry = ProcessRegistry()

    def teardown_method(self):
        """清理后台线程，避免 daemon 线程累积导致死锁。"""
        for sid in list(self.registry._running):
            self.registry._running[sid].exited = True
            self.registry._move_to_finished(self.registry._running[sid])
        self.registry._running.clear()
        self.registry._finished.clear()

    def test_spawn_via_env(self):
        """通过 env 接口启动后台进程。"""
        env = MagicMock()
        env.execute.return_value = _MOCK_ENV_RESULT
        session = self.registry.spawn(env, "echo hello", cwd="/tmp")
        assert session.id.startswith("proc_")
        assert session.command == "echo hello"
        assert session.cwd == "/tmp"
        assert session.pid_scope == "sandbox"

    def test_spawn_local_popen(self):
        """通过本地 Popen 启动后台进程。"""
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        # 让 buffer.read1 返回 b""（EOF），让 _reader_loop 自然退出
        mock_stdout = MagicMock()
        mock_stdout.buffer.read1.return_value = b""
        mock_stdout.read.return_value = ""
        mock_proc.stdout = mock_stdout
        with patch("kocor.tools.toolsets.bash.process_registry.subprocess.Popen",
                   return_value=mock_proc) as mock_popen:
            session = self.registry.spawn_local("echo hello")
            assert session.id.startswith("proc_")
            assert session.command == "echo hello"
            assert session.pid == 99999
            mock_popen.assert_called_once()

    def test_poll_returns_not_found(self):
        result = self.registry.poll("nonexistent")
        assert result["status"] == "not_found"

    def test_kill_returns_not_found(self):
        result = self.registry.kill("nonexistent")
        assert result["status"] == "not_found"

    def test_list_sessions_empty(self):
        assert self.registry.list_sessions() == []

    def test_spawn_and_list(self):
        env = MagicMock()
        env.execute.return_value = _MOCK_ENV_RESULT
        session = self.registry.spawn(env, "echo hello")
        sessions = self.registry.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session.id

    def test_read_log(self):
        session = ProcessSession(id="proc_test", command="echo hello")
        session.output_buffer = "line1\nline2\nline3\n"
        self.registry._running["proc_test"] = session
        result = self.registry.read_log("proc_test", limit=2)
        assert result["status"] == "running"
        assert "line2" in result["output"] or "line3" in result["output"]

    def test_output_buffer_rolling(self):
        """验证输出缓冲区滚动。"""
        session = ProcessSession(id="proc_test", command="echo hello", max_output_chars=100)
        session.output_buffer = "x" * 50
        # 追加更多输出，触发滚动
        session.output_buffer += "y" * 60
        if len(session.output_buffer) > session.max_output_chars:
            session.output_buffer = session.output_buffer[-session.max_output_chars:]
        assert len(session.output_buffer) <= 100
        # 最后 100 个字符包含后面追加的 y
        assert "y" in session.output_buffer

    def test_completion_queue(self):
        """验证完成通知队列。"""
        self.registry._notify_completion("proc_test", "echo hello", 0)
        assert not self.registry.completion_queue.empty()
        evt = self.registry.completion_queue.get_nowait()
        assert evt["type"] == "completion"
        assert evt["session_id"] == "proc_test"

    def test_cleanup_finished_processes(self):
        """验证已结束进程清理。"""
        old = ProcessSession(
            id="proc_old",
            command="echo old",
            started_at=time.time() - 2000,  # 超过 30 分钟 TTL
            exited=True,
            exit_code=0,
        )
        self.registry._finished["proc_old"] = old
        self.registry._prune_if_needed()
        assert "proc_old" not in self.registry._finished


class TestProcessRegistryIntegration:
    """ProcessRegistry 与 ProcessTool 集成测试。"""

    def setup_method(self):
        self.registry = ProcessRegistry()

    def teardown_method(self):
        """清理后台线程。"""
        for sid in list(self.registry._running):
            self.registry._running[sid].exited = True
            self.registry._move_to_finished(self.registry._running[sid])
        self.registry._running.clear()
        self.registry._finished.clear()

    def test_poll_after_spawn(self):
        env = MagicMock()
        env.execute.return_value = _MOCK_ENV_RESULT
        session = self.registry.spawn(env, "sleep 10")
        result = self.registry.poll(session.id)
        assert result["status"] in ("running", "exited")
        assert result["session_id"] == session.id