"""cron worker 子进程生命周期管理器（主进程侧）。

职责：启动 / 停止独立的 cron worker 子进程。tick 轮询与作业执行
全部在子进程内完成，主进程不参与。

停止协议（跨平台，无信号依赖）：
1. 关闭子进程 stdin → 子进程 stdin watcher 收到 EOF → 自行优雅退出。
2. 若子进程未在限时内退出，terminate() 强制中断。
3. 仍无响应则 kill() 兜底。
"""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# 子进程入口模块：`python -m kocor.cron.cron_worker`
_WORKER_MODULE = "kocor.cron.cron_worker"


class CronWorkerProcess:
    """cron worker 子进程生命周期管理器。"""

    # 停止等待限时（秒）
    GRACEFUL_WAIT = 10
    TERMINATE_WAIT = 5

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @property
    def is_running(self) -> bool:
        """子进程是否仍在运行。"""
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """启动 cron worker 子进程。已运行时为幂等 no-op。"""
        if self.is_running:
            return
        # stdin=PIPE：主进程关闭它即向子进程发 EOF 停止信号
        self._proc = subprocess.Popen(
            [sys.executable, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE,
        )
        logger.info("cron worker 子进程已启动 (pid=%s)", self._proc.pid)

    def stop(self) -> None:
        """停止 cron worker 子进程：stdin EOF → terminate → kill。"""
        if self._proc is None:
            return

        # 1) 优雅：关闭 stdin → 子进程 watcher 收到 EOF → 自行退出
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except OSError as e:
            logger.warning("关闭 cron worker stdin 失败: %s", e)

        try:
            self._proc.wait(timeout=self.GRACEFUL_WAIT)
        except subprocess.TimeoutExpired:
            # 2) 兜底一：强制终止
            logger.warning("cron worker 未在 %ds 内退出，发送 terminate", self.GRACEFUL_WAIT)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=self.TERMINATE_WAIT)
            except subprocess.TimeoutExpired:
                # 3) 兜底二：kill
                logger.warning("cron worker 未响应 terminate，发送 kill")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=self.TERMINATE_WAIT)
                except subprocess.TimeoutExpired:
                    logger.error("cron worker 未响应 kill，放弃等待")

        self._proc = None
        logger.info("cron worker 子进程已停止")
