"""测试 cron worker 子进程生命周期管理器（主进程侧）。

仅验证进程启停逻辑，不真实 spawn 子进程（用 mock Popen）。
真实子进程行为由 test_cron_worker.py 端到端覆盖。
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from kocor.tools.toolsets.cron.worker_process import CronWorkerProcess


class TestCronWorkerProcessStart:
    """启动逻辑。"""

    def test_start_spawns_subprocess(self):
        """start() 以 `python -m kocor.cron_worker` spawn 子进程，stdin 接管道。"""
        proc = MagicMock()
        proc.poll.return_value = None  # 运行中
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()

        args, kwargs = popen.call_args
        assert args[0] == [sys.executable, "-m", "kocor.cron_worker"]
        assert kwargs["stdin"] == subprocess.PIPE

    def test_start_idempotent_when_running(self):
        """已运行时重复 start 不再 spawn。"""
        proc = MagicMock()
        proc.poll.return_value = None
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            wp.start()
            assert popen.call_count == 1

    def test_is_running_true_after_start(self):
        proc = MagicMock()
        proc.poll.return_value = None
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            assert not wp.is_running
            wp.start()
            assert wp.is_running

    def test_is_running_false_after_proc_exited(self):
        proc = MagicMock()
        proc.poll.return_value = 0  # 已退出
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            assert not wp.is_running


class TestCronWorkerProcessStop:
    """停止逻辑：stdin EOF 优雅退出 + terminate/kill 兜底。"""

    def test_stop_closes_stdin_then_waits(self):
        """stop() 先关 stdin（发 EOF），再 wait 收尾。"""
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait.return_value = 0
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            wp.stop()

        proc.stdin.close.assert_called_once()
        proc.wait.assert_called_once()
        assert proc.terminate.call_count == 0
        assert proc.kill.call_count == 0

    def test_stop_falls_back_to_terminate_on_timeout(self):
        """wait 超时 → terminate → 再次 wait 成功。"""
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=10), 0]
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            wp.stop()

        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        assert proc.wait.call_count == 2

    def test_stop_falls_back_to_kill_on_second_timeout(self):
        """terminate 后 wait 仍超时 → kill。"""
        proc = MagicMock()
        proc.stdin = MagicMock()
        # graceful wait / terminate-then-wait 均超时，kill 后 wait 返回 0
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="x", timeout=10),
            subprocess.TimeoutExpired(cmd="x", timeout=5),
            0,
        ]
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            wp.stop()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_stop_noop_when_not_started(self):
        """未启动时 stop() 不报错。"""
        wp = CronWorkerProcess()
        wp.stop()  # 不应抛异常

    def test_stop_clears_handle(self):
        """stop 后 is_running 为 False 且可重新 start。"""
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait.return_value = 0
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            wp.stop()
            assert not wp.is_running
            wp.start()
            assert popen.call_count == 2

    def test_stop_tolerates_stdin_close_error(self):
        """stdin.close() 抛异常时仍继续 wait。"""
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.close.side_effect = OSError("broken pipe")
        proc.wait.return_value = 0
        with patch("kocor.tools.toolsets.cron.worker_process.subprocess.Popen") as popen:
            popen.return_value = proc
            wp = CronWorkerProcess()
            wp.start()
            wp.stop()  # 不应抛异常

        proc.wait.assert_called_once()
