"""端到端测试 cron worker 子进程生命周期。

cron_worker 是独立子进程，需要 LLM 配置才能完全启动。
本文件包含：
1. 导入验证
2. 条件 spawn 测试（运行时需有 LLM 配置）
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cron_worker_module_importable():
    """cron_worker 模块可导入且 main 函数存在。"""
    from kocor.cron.cron_worker import main

    assert callable(main)


@pytest.mark.skipif(
    not (
        (Path(".env").exists())
        or "OPENAI_API_KEY" in os.environ
        or "ANTHROPIC_API_KEY" in os.environ
    ),
    reason="需要 LLM 配置（.env、OPENAI_API_KEY 或 ANTHROPIC_API_KEY）才能 spawn 子进程",
)
def test_worker_spawn_exits_on_stdin_eof():
    """spawn cron_worker，关闭 stdin 后子进程应正常退出。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "kocor.cron.cron_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 关闭 stdin → 子进程 watcher 收到 EOF → 自行退出
    proc.stdin.close()

    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("cron_worker 未在 20s 内退出")

    assert proc.returncode == 0, (
        f"cron_worker exit code {proc.returncode}, stderr={stderr.decode()}"
    )