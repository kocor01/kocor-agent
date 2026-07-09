"""测试 cron 调度器。"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kocor.tools.toolsets.cron.jobs import (
    HAS_CRONITER,
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    load_jobs,
    mark_job_run,
    save_jobs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cron_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cron"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def patch_cron_dir(monkeypatch, cron_dir: Path):
    import kocor.tools.toolsets.cron.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_module, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_module, "OUTPUT_DIR", cron_dir / "output")
    jobs_module._job_id_counter = 0
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCronScheduler:
    """测试 CronScheduler 生命周期和基本功能。"""

    def test_start_stop(self):
        """启动和停止调度器。"""
        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        scheduler = CronScheduler(tick_interval=0.1)
        assert not scheduler.is_running

        scheduler.start()
        assert scheduler.is_running

        scheduler.stop()
        assert not scheduler.is_running

    def test_double_start(self):
        """重复启动无效。"""
        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        scheduler = CronScheduler(tick_interval=0.1)
        scheduler.start()
        scheduler.start()  # 不应报错
        assert scheduler.is_running
        scheduler.stop()

    def test_double_stop(self):
        """重复停止无效。"""
        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        scheduler = CronScheduler(tick_interval=0.1)
        scheduler.start()
        scheduler.stop()
        scheduler.stop()  # 不应报错
        assert not scheduler.is_running

    def test_tick_processes_due_jobs(self):
        """tick 处理到期作业。"""
        # 创建间隔作业，手动设置 next_run_at 到过去
        job = create_job(prompt="test job", schedule="*/10 * * * *")
        job_id = job["id"]

        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        # 将 next_run_at 设为已过期
        import kocor.tools.toolsets.cron.jobs as jobs_module
        from datetime import datetime, timedelta

        jobs = jobs_module.load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["next_run_at"] = (datetime.now().astimezone() - timedelta(seconds=10)).isoformat()
        jobs_module.save_jobs(jobs)

        # 执行 tick
        scheduler = CronScheduler(tick_interval=0.1)
        scheduler._tick()

        # 验证作业被标记为已执行
        refreshed = get_job(job_id)
        assert refreshed is not None
        assert refreshed.get("last_status") == "ok"

    def test_claim_job_prevents_duplicate(self):
        """claim_job_for_fire 防止重复执行。"""
        job = create_job(prompt="dup test", schedule="*/10 * * * *")
        job_id = job["id"]

        first = claim_job_for_fire(job_id)
        assert first is True

        second = claim_job_for_fire(job_id)
        assert second is False

    def test_get_due_jobs_returns_due(self):
        """get_due_jobs 返回到期作业。"""
        job = create_job(prompt="due test", schedule="*/10 * * * *")

        from kocor.tools.toolsets.cron.jobs import get_due_jobs, load_jobs

        # 将 next_run_at 设为过去
        import kocor.tools.toolsets.cron.jobs as jobs_module
        from datetime import datetime, timedelta

        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["next_run_at"] = (datetime.now().astimezone() - timedelta(seconds=5)).isoformat()
        jobs_module.save_jobs(jobs)

        due = get_due_jobs()
        ids = [j["id"] for j in due]
        assert job["id"] in ids

    def test_get_due_jobs_empty_when_no_due(self):
        """无到期作业时返回空列表。"""
        from kocor.tools.toolsets.cron.jobs import get_due_jobs

        create_job(prompt="future job", schedule="*/10 * * * *")
        due = get_due_jobs()
        assert len(due) == 0  # next_run_at 在将来

    def test_mark_job_run_tracks_status(self):
        """mark_job_run 更新作业状态。"""
        job = create_job(prompt="track test", schedule="*/10 * * * *")
        job_id = job["id"]

        from kocor.tools.toolsets.cron.jobs import mark_job_run, get_job

        mark_job_run(job_id, success=True)
        refreshed = get_job(job_id)
        assert refreshed["last_status"] == "ok"
        assert refreshed["last_run_at"] is not None
        assert refreshed["repeat"]["completed"] == 1


class TestCronSchedulerEdgeCases:
    """边界情况测试。"""

    def test_daemon_thread_does_not_block_exit(self):
        """调度器线程是守护线程。"""
        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        scheduler = CronScheduler(tick_interval=10)
        scheduler.start()
        assert scheduler._tick_thread is not None
        assert scheduler._tick_thread.daemon is True
        scheduler.stop()

    def test_tick_no_due_jobs_no_error(self):
        """无到期作业时 tick 不报错。"""
        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        scheduler = CronScheduler(tick_interval=0.1)
        scheduler._tick()  # 不应抛出异常
        scheduler.stop()

    def test_cron_disabled_toolsets(self):
        """验证 cron 禁用的工具集列表。"""
        from kocor.tools.toolsets.cron.types import DISABLED_TOOLSETS_IN_CRON

        assert "cronjob" in DISABLED_TOOLSETS_IN_CRON