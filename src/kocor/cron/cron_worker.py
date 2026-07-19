"""cron worker 子进程入口。

由主进程（CLI 层 CronWorkerProcess）以 `python -m kocor.cron.cron_worker` spawn。

停止协议：
  主进程关闭其 stdin 管道末端 → 本进程 `sys.stdin.read()` 收到 EOF →
  停止 CronScheduler → 退出。全程无信号依赖，跨平台可靠。
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """cron worker 子进程主函数。"""
    from kocor.cron.agent_builder import build_cron_agent

    agent, scheduler = build_cron_agent()
    scheduler.start()
    logger.info("cron worker 子进程已就绪 (pid=%s)", os.getpid())

    try:
        # 阻塞等待 stdin EOF：主进程关闭管道时投递 EOF，
        # 作为跨平台停止信号（不依赖 SIGTERM/SIGINT，后者在 Windows 不适用）。
        sys.stdin.read()
    except (OSError, ValueError, KeyboardInterrupt):
        # stdin 异常关闭（竞态、管道损坏等）或 Ctrl+C 传递到子进程时视为停止信号。
        pass
    finally:
        logger.info("cron worker 子进程正在退出...")
        scheduler.stop()


if __name__ == "__main__":
    main()