"""测试 cron 作业 CRUD 操作。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kocor.tools.toolset.cron.jobs import (
    AmbiguousJobReference,
    create_job,
    get_job,
    list_jobs,
    load_jobs,
    pause_job,
    remove_job,
    resolve_job_ref,
    resume_job,
    save_jobs,
    update_job,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cron_dir(tmp_path: Path) -> Path:
    """创建临时 cron 目录，模拟 ~/.kocor/cron/。"""
    d = tmp_path / "cron"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def patch_cron_dir(monkeypatch, cron_dir: Path):
    """将 CRON_DIR 指向临时目录。"""
    import kocor.tools.toolset.cron.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_module, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_module, "OUTPUT_DIR", cron_dir / "output")
    yield
    # 清理类级别的作业 ID 计数器
    jobs_module._job_id_counter = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJobsCRUD:
    """测试作业 CRUD 的基本操作。"""

    def test_create_job_minimal(self):
        """创建最小配置的作业（仅 prompt + schedule）。"""
        job = create_job(prompt="say hello", schedule="30m")
        assert job["id"] is not None
        assert job["prompt"] == "say hello"
        assert job["schedule"]["kind"] == "once"
        assert job["enabled"] is True
        assert job["state"] == "scheduled"
        assert job["next_run_at"] is not None

    def test_create_job_every_interval(self):
        """创建循环间隔作业。"""
        job = create_job(prompt="check status", schedule="every 10m")
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 10
        assert job["repeat"] == {"times": None, "completed": 0}

    def test_create_job_with_name(self):
        """创建命名作业。"""
        job = create_job(prompt="daily report", schedule="every 1h", name="Daily Report")
        assert job["name"] == "Daily Report"

    def test_create_job_name_defaults_to_prompt(self):
        """未指定 name 时默认截取 prompt 前 50 字符。"""
        job = create_job(prompt="say hello", schedule="30m")
        assert job["name"] == "say hello"

    def test_create_job_auto_repeat_once_for_oneshot(self):
        """一次性调度自动设置 repeat=1。"""
        job = create_job(prompt="one time", schedule="30m")
        assert job["repeat"] == {"times": 1, "completed": 0}

    def test_create_job_with_repeat(self):
        """指定重复次数。"""
        job = create_job(prompt="run 5 times", schedule="every 10m", repeat=5)
        assert job["repeat"] == {"times": 5, "completed": 0}

    def test_get_job_by_id(self):
        """按 ID 获取作业。"""
        created = create_job(prompt="test", schedule="30m")
        fetched = get_job(created["id"])
        assert fetched is not None
        assert fetched["id"] == created["id"]
        assert fetched["prompt"] == "test"

    def test_get_job_not_found(self):
        """不存在的 ID 返回 None。"""
        assert get_job("nonexistent") is None

    def test_resolve_job_ref_by_id(self):
        """按 ID 解析引用。"""
        created = create_job(prompt="test", schedule="30m")
        resolved = resolve_job_ref(created["id"])
        assert resolved is not None
        assert resolved["id"] == created["id"]

    def test_resolve_job_ref_by_name(self):
        """按名称解析引用。"""
        created = create_job(prompt="test name resolve", schedule="30m", name="UniqueName")
        resolved = resolve_job_ref("UniqueName")
        assert resolved is not None
        assert resolved["id"] == created["id"]

    def test_resolve_job_ref_not_found(self):
        """不存在的引用返回 None。"""
        assert resolve_job_ref("nonexistent") is None

    def test_resolve_job_ref_ambiguous(self):
        """模糊名称抛出 AmbiguousJobReference。"""
        create_job(prompt="first", schedule="30m", name="SameName")
        create_job(prompt="second", schedule="30m", name="SameName")
        with pytest.raises(AmbiguousJobReference) as exc:
            resolve_job_ref("SameName")
        assert "ambiguous" in str(exc.value).lower()

    def test_list_jobs(self):
        """列出所有作业。"""
        create_job(prompt="job1", schedule="30m")
        create_job(prompt="job2", schedule="every 10m")
        jobs = list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_exclude_disabled(self):
        """list_jobs 默认排除已禁用作业。"""
        job = create_job(prompt="to disable", schedule="30m")
        update_job(job["id"], {"enabled": False, "state": "paused"})
        jobs = list_jobs()
        assert len(jobs) == 0

    def test_list_jobs_include_disabled(self):
        """include_disabled=True 包含已禁用作业。"""
        job = create_job(prompt="disabled job", schedule="30m")
        update_job(job["id"], {"enabled": False, "state": "paused"})
        jobs = list_jobs(include_disabled=True)
        assert len(jobs) == 1

    def test_remove_job(self):
        """删除作业。"""
        created = create_job(prompt="to delete", schedule="30m")
        assert remove_job(created["id"]) is True
        assert get_job(created["id"]) is None

    def test_remove_job_not_found(self):
        """删除不存在的作业返回 False。"""
        assert remove_job("nonexistent") is False

    def test_update_job_prompt(self):
        """更新作业 prompt。"""
        job = create_job(prompt="original", schedule="30m")
        updated = update_job(job["id"], {"prompt": "updated"})
        assert updated["prompt"] == "updated"

    def test_update_job_name(self):
        """更新作业名称。"""
        job = create_job(prompt="test", schedule="30m")
        updated = update_job(job["id"], {"name": "NewName"})
        assert updated["name"] == "NewName"

    def test_update_job_schedule(self):
        """更新调度计划。"""
        job = create_job(prompt="test", schedule="30m")
        updated = update_job(job["id"], {"schedule": "every 1h"})
        assert updated["schedule"]["kind"] == "interval"
        assert updated["schedule"]["minutes"] == 60

    def test_update_job_not_found(self):
        """更新不存在的作业返回 None。"""
        assert update_job("nonexistent", {"prompt": "new"}) is None

    def test_pause_job(self):
        """暂停作业。"""
        job = create_job(prompt="test", schedule="every 10m")
        paused = pause_job(job["id"])
        assert paused["enabled"] is False
        assert paused["state"] == "paused"
        assert paused["paused_at"] is not None

    def test_pause_job_not_found(self):
        """暂停不存在的作业返回 None。"""
        assert pause_job("nonexistent") is None

    def test_resume_job(self):
        """恢复暂停的作业。"""
        job = create_job(prompt="test", schedule="every 10m")
        pause_job(job["id"])
        resumed = resume_job(job["id"])
        assert resumed["enabled"] is True
        assert resumed["state"] == "scheduled"
        assert resumed["next_run_at"] is not None

    def test_resume_job_not_found(self):
        """恢复不存在的作业返回 None。"""
        assert resume_job("nonexistent") is None


class TestJobsStorage:
    """测试作业的持久化存储。"""

    def test_persistence_across_load(self, cron_dir: Path):
        """创建后文件写入磁盘，可重新加载。"""
        job = create_job(prompt="persist test", schedule="30m")
        jobs_file = cron_dir / "jobs.json"
        assert jobs_file.exists()

        loaded = load_jobs()
        ids = [j["id"] for j in loaded]
        assert job["id"] in ids

    def test_jobs_file_structure(self, cron_dir: Path):
        """验证 JSON 文件结构。"""
        create_job(prompt="test", schedule="30m")
        jobs_file = cron_dir / "jobs.json"
        with open(jobs_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "jobs" in data
        assert "updated_at" in data
        assert len(data["jobs"]) == 1

    def test_empty_jobs_file(self, cron_dir: Path):
        """空或不存在的文件返回空列表。"""
        assert load_jobs() == []

    def test_remove_cleans_output_dir(self, cron_dir: Path):
        """删除作业同时清理输出目录。"""
        job = create_job(prompt="cleanup test", schedule="30m")
        output_dir = cron_dir / "output" / job["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.md").write_text("output", encoding="utf-8")

        remove_job(job["id"])
        assert not output_dir.exists()


class TestJobsEdgeCases:
    """边界情况测试。"""

    def test_create_job_with_empty_prompt_fails(self):
        """空的 prompt 和 skills 应报错。"""
        with pytest.raises(ValueError, match="prompt or at least one skill"):
            create_job(prompt="", schedule="30m")

    def test_create_job_with_skills_only(self):
        """仅 skills 也能创建作业。"""
        job = create_job(prompt="", schedule="every 10m", skills=["greeter"])
        assert job["skills"] == ["greeter"]

    def test_create_job_with_zero_repeat(self):
        """repeat=0 视为无限（None）。"""
        job = create_job(prompt="test", schedule="every 10m", repeat=0)
        assert job["repeat"]["times"] is None

    def test_create_job_with_negative_repeat(self):
        """repeat 负数视为无限。"""
        job = create_job(prompt="test", schedule="every 10m", repeat=-1)
        assert job["repeat"]["times"] is None

    def test_resolve_job_ref_empty_string(self):
        """空字符串解析返回 None。"""
        assert resolve_job_ref("") is None

    def test_resolve_job_ref_none(self):
        """None 解析返回 None。"""
        assert resolve_job_ref(None) is None  # type: ignore[arg-type]

    def test_update_job_immutable_id(self):
        """不能更新作业 ID。"""
        job = create_job(prompt="test", schedule="30m")
        with pytest.raises(ValueError, match="cannot be updated"):
            update_job(job["id"], {"id": "new-id"})