"""Cron 作业存储与 CRUD 操作。

作业存储在 ~/.kocor/cron/jobs.json，输出保存在 ~/.kocor/cron/output/{job_id}/。
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from kocor.config import Config
from kocor.tools.toolsets.cron.types import (
    CRON_DIR_NAME,
    JOBS_FILE_NAME,
    JOBS_LOCK_NAME,
    ONESHOT_GRACE_SECONDS,
    OUTPUT_DIR_NAME,
    STATE_COMPLETED,
    STATE_ERROR,
    STATE_PAUSED,
    STATE_SCHEDULED,
)

logger = logging.getLogger(__name__)

# 跨进程文件锁（Unix fcntl / Windows msvcrt）
try:
    import fcntl
except ImportError:
    fcntl = None
try:
    import msvcrt
except ImportError:
    msvcrt = None

# 尝试导入 croniter（可选，用于 cron 表达式解析）
try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False
    croniter = None  # type: ignore[assignment]

# =============================================================================
# 路径配置
# =============================================================================

# 根据 data_dir 或默认值确定 cron 存储根目录
def _resolve_cron_root() -> Path:
    """解析 cron 存储根目录。"""
    data_dir = Config.load().memory_dir  # 复用 memory_dir 的基目录逻辑
    base = Path(data_dir).parent if data_dir else Path.home() / ".kocor"
    return base.resolve() / CRON_DIR_NAME


CRON_DIR = _resolve_cron_root()
JOBS_FILE = CRON_DIR / JOBS_FILE_NAME
OUTPUT_DIR = CRON_DIR / OUTPUT_DIR_NAME

# 线程锁保护 load→modify→save 临界区
_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()

# 测试用计数器（测试时重置）
_job_id_counter = 0

# 不可变字段
_IMMUTABLE_JOB_FIELDS = frozenset({"id"})


# =============================================================================
# 工具函数
# =============================================================================


def _jobs_lock_file() -> Path:
    """返回跨进程文件锁路径。"""
    return CRON_DIR / JOBS_LOCK_NAME


@contextlib.contextmanager
def _jobs_lock():
    """序列化 load_jobs→modify→save_jobs 临界区。

    组合线程锁（RLock）和跨进程文件锁，确保并发安全。
    """
    depth = getattr(_jobs_lock_state, "depth", 0)
    if depth:
        _jobs_lock_state.depth = depth + 1
        try:
            yield
        finally:
            _jobs_lock_state.depth -= 1
        return

    with _jobs_file_lock:
        _jobs_lock_state.depth = 1
        try:
            with _acquire_file_lock():
                yield
        finally:
            _jobs_lock_state.depth = 0


@contextlib.contextmanager
def _acquire_file_lock():
    """获取跨进程文件锁（失败时降级为仅线程锁）。"""
    lock_fd = None
    try:
        try:
            ensure_dirs()
            lock_fd = open(_jobs_lock_file(), "a+", encoding="utf-8")
            lock_fd.seek(0)
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            elif msvcrt is not None:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)
        except (OSError, IOError) as e:
            logger.warning("跨进程文件锁不可用 (%s)，仅使用线程锁", e)
        yield
    finally:
        if lock_fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                elif msvcrt is not None:
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
            finally:
                lock_fd.close()


def _secure_dir(path: Path) -> None:
    """设置目录权限为 owner-only（Unix only）。"""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path) -> None:
    """设置文件权限为 owner-only（Unix only）。"""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def ensure_dirs() -> None:
    """确保 cron 目录存在并设置安全权限。"""
    CRON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _secure_dir(CRON_DIR)
    _secure_dir(OUTPUT_DIR)


def _now() -> datetime:
    """获取当前时间（带时区）。"""
    return datetime.now().astimezone()


def _generate_job_id() -> str:
    """生成短作业 ID。"""
    global _job_id_counter
    _job_id_counter += 1
    return uuid.uuid4().hex[:12]


# =============================================================================
# 调度解析
# =============================================================================


def parse_schedule(schedule: str) -> dict[str, Any]:
    """解析调度字符串为结构化格式。

    返回 dict 包含：
        - kind: "once" | "cron"
        - display: 显示用字符串
        - 各类别特定字段

    支持的格式：
        "2 22 * * *"       → cron 表达式（覆盖所有重复场景）
        "2026-07-08T14:00:00" → ISO 时间戳（单次定时）
    """
    schedule = schedule.strip()
    original = schedule

    if not schedule:
        raise ValueError("调度计划不能为空。")

    # cron 表达式（5 个空格分隔的字段）
    parts = schedule.split()
    if len(parts) >= 5 and all(
        re.match(r'^[\d\*\-,/]+$', p) for p in parts[:5]
    ):
        if not HAS_CRONITER:
            raise ValueError(
                "Cron expressions require 'croniter' package. Install with: pip install croniter"
            )
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"无效的 cron 表达式 '{schedule}': {e}")
        return {
            "kind": "cron",
            "expr": schedule,
            "display": schedule,
        }

    # ISO 时间戳（包含 T 或日期格式）
    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_now().tzinfo)
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}",
            }
        except ValueError as e:
            raise ValueError(f"无效的时间戳 '{schedule}': {e}")

    raise ValueError(
        f"无效的调度计划 '{original}'。请使用:\n"
        f"  - Cron 表达式: '2 22 * * *'（循环，每天 22:02）\n"
        f"  - 时间戳: '2026-07-08T22:00:00'（单次定时）"
    )


def compute_next_run(schedule: dict[str, Any], last_run_at: str | None = None) -> str | None:
    """计算下次执行时间。

    返回 ISO 时间戳字符串，若无下次执行则返回 None。
    """
    now = _now()

    kind = schedule.get("kind")

    if kind == "once":
        run_at = schedule.get("run_at")
        if not run_at:
            return None
        run_at_dt = datetime.fromisoformat(run_at)
        if run_at_dt.tzinfo is None:
            run_at_dt = run_at_dt.replace(tzinfo=now.tzinfo)
        if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
            return run_at
        # 已执行过则不再返回
        if last_run_at:
            return None
        return None

    if kind == "interval":
        minutes = schedule.get("minutes", 0)
        if minutes <= 0:
            return None
        if last_run_at:
            last = datetime.fromisoformat(last_run_at)
            if last.tzinfo is None:
                last = last.replace(tzinfo=now.tzinfo)
            next_run = last + timedelta(minutes=minutes)
        else:
            next_run = now + timedelta(minutes=minutes)
        return next_run.isoformat()

    if kind == "cron":
        if not HAS_CRONITER:
            logger.warning("croniter 未安装，无法计算 cron 调度的时间。")
            return None
        expr = schedule.get("expr", "")
        base_time = now
        if last_run_at:
            base_time = datetime.fromisoformat(last_run_at)
        try:
            cron = croniter(expr, base_time)
            next_run = cron.get_next(datetime)
            return next_run.isoformat()
        except Exception:
            return None

    return None


# =============================================================================
# 作业规范化
# =============================================================================


def _normalize_skill_list(skill: str | None = None, skills: list[str] | None = None) -> list[str]:
    """规范化技能列表。"""
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)
    normalized: list[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_job_text(value: Any, fallback: str = "") -> str:
    """安全转换为字符串，空值返回 fallback。"""
    if value is None:
        return fallback
    return str(value)


def _normalize_job_record(job: dict[str, Any]) -> dict[str, Any]:
    """返回只读安全的作业记录（兼容旧格式的字段缺失）。"""
    normalized = dict(job)
    job_id = _normalize_job_text(normalized.get("id"), "unknown")
    prompt = _normalize_job_text(normalized.get("prompt"))
    normalized["id"] = job_id
    normalized["prompt"] = prompt

    name = _normalize_job_text(normalized.get("name")).strip()
    if not name:
        skills = _normalize_skill_list(normalized.get("skill"), normalized.get("skills"))
        label_source = (
            prompt
            or (skills[0] if skills else "")
            or normalized.get("script", "")
            or job_id
            or "cron job"
        )
        name = label_source[:50].strip() or "cron job"
    normalized["name"] = name

    schedule_display = _normalize_job_text(normalized.get("schedule_display")).strip()
    if not schedule_display:
        schedule = normalized.get("schedule", {})
        if isinstance(schedule, dict):
            for key in ("display", "value", "expr", "run_at"):
                text = _normalize_job_text(schedule.get(key)).strip()
                if text:
                    schedule_display = text
                    break
        elif schedule is not None:
            schedule_display = str(schedule)
        else:
            schedule_display = "?"
    normalized["schedule_display"] = schedule_display

    state = _normalize_job_text(normalized.get("state")).strip()
    if not state:
        state = STATE_SCHEDULED if normalized.get("enabled", True) else STATE_PAUSED
    normalized["state"] = state

    return normalized


# =============================================================================
# 作业 CRUD
# =============================================================================


def load_jobs() -> list[dict[str, Any]]:
    """从存储加载所有作业。"""
    ensure_dirs()
    if not JOBS_FILE.exists():
        return []

    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                data = json.loads(f.read(), strict=False)
        except Exception as e:
            raise RuntimeError(f"Cron 数据库损坏且无法修复: {e}") from e
    except IOError as e:
        raise RuntimeError(f"读取 cron 数据库失败: {e}") from e

    if isinstance(data, dict):
        return data.get("jobs", [])
    if isinstance(data, list):
        if data:
            save_jobs(data)
        return data
    raise RuntimeError(
        f"Cron 数据库损坏: 期望 {{'jobs': [...]}}, 实际为 {type(data).__name__}"
    )


def _save_jobs_unlocked(jobs: list[dict[str, Any]]) -> None:
    """保存作业到存储（调用者必须持有 _jobs_lock）。"""
    ensure_dirs()
    fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_FILE.parent), suffix=".tmp", prefix=".jobs_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"jobs": jobs, "updated_at": _now().isoformat()}, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, JOBS_FILE)
        _secure_file(JOBS_FILE)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_jobs(jobs: list[dict[str, Any]]) -> None:
    """保存作业到存储。"""
    with _jobs_lock():
        _save_jobs_unlocked(jobs)


def _job_output_dir(job_id: str) -> Path:
    """解析作业的输出目录，拒绝路径逃逸。"""
    text = str(job_id or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"无效的 cron job id: {job_id!r}")
    return OUTPUT_DIR / text


def create_job(
    prompt: str | None = None,
    schedule: str | None = None,
    name: str | None = None,
    repeat: int | None = None,
    skill: str | None = None,
    skills: list[str] | None = None,
    script: str | None = None,
    no_agent: bool = False,
    context_from: list[str] | None = None,
    enabled_toolsets: list[str] | None = None,
    workdir: str | None = None,
) -> dict[str, Any]:
    """创建新 cron 作业。

    Args:
        prompt: 提示词（no_agent 模式可选）
        schedule: 调度计划字符串
        name: 可选名称
        repeat: 重复次数（None=无限，1=一次性）
        skill: 旧版单技能名称
        skills: 技能列表
        script: 脚本路径
        no_agent: True=仅执行脚本
        context_from: 上游作业 ID 列表
        enabled_toolsets: 限制工具集
        workdir: 工作目录

    Returns:
        创建的作业 dict

    Raises:
        ValueError: 参数校验失败
    """
    if not schedule:
        raise ValueError("schedule is required for create.")

    parsed_schedule = parse_schedule(schedule)

    # 校验 prompt 或 skills 至少一个
    normalized_skills = _normalize_skill_list(skill, skills)
    prompt_text = _normalize_job_text(prompt)
    if not no_agent and not prompt_text and not normalized_skills:
        raise ValueError("create requires either prompt or at least one skill.")

    # 规范化 repeat
    if repeat is not None and repeat <= 0:
        repeat = None

    # 一次性调度自动设置 repeat=1
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1

    now = _now().isoformat()
    job_id = _generate_job_id()

    next_run_at = compute_next_run(parsed_schedule)

    label_source = (
        prompt_text
        or (normalized_skills[0] if normalized_skills else None)
        or (script if no_agent else None)
        or "cron job"
    )

    job = {
        "id": job_id,
        "name": name or label_source[:50].strip(),
        "prompt": prompt_text,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "script": script,
        "no_agent": bool(no_agent),
        "context_from": context_from or None,
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {"times": repeat, "completed": 0},
        "enabled": True,
        "state": STATE_SCHEDULED,
        "paused_at": None,
        "paused_reason": None,
        "created_at": now,
        "next_run_at": next_run_at,
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "enabled_toolsets": enabled_toolsets or None,
        "workdir": workdir or None,
    }

    with _jobs_lock():
        jobs = load_jobs()
        jobs.append(job)
        save_jobs(jobs)

    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    """按 ID 获取作业。"""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            return _normalize_job_record(job)
    return None


class AmbiguousJobReference(LookupError):
    """当作业名称匹配多个作业时抛出。"""

    def __init__(self, ref: str, matches: list[dict[str, Any]]):
        self.ref = ref
        self.matches = matches
        ids = ", ".join(m["id"] for m in matches)
        super().__init__(
            f"Job name '{ref}' is ambiguous — matches {len(matches)} jobs: {ids}. "
            f"Use the job ID instead."
        )


def resolve_job_ref(ref: str | None) -> dict[str, Any] | None:
    """解析作业引用（ID 或名称）。

    - 精确 ID 匹配优先
    - 其次大小写不敏感的名称匹配
    - 名称匹配多个作业时抛出 AmbiguousJobReference
    """
    if not ref:
        return None
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == ref:
            return _normalize_job_record(job)
    ref_lower = ref.lower()
    name_matches = [j for j in jobs if (j.get("name") or "").lower() == ref_lower]
    if not name_matches:
        return None
    if len(name_matches) > 1:
        raise AmbiguousJobReference(
            ref, [_normalize_job_record(j) for j in name_matches]
        )
    return _normalize_job_record(name_matches[0])


def list_jobs(include_disabled: bool = False) -> list[dict[str, Any]]:
    """列出所有作业，可选包含已禁用的。"""
    jobs = [_normalize_job_record(j) for j in load_jobs()]
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    return jobs


def update_job(job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """更新作业字段。

    Args:
        job_id: 作业 ID
        updates: 要更新的字段字典

    Returns:
        更新后的作业，未找到返回 None

    Raises:
        ValueError: 更新不可变字段或参数校验失败
    """
    bad_fields = _IMMUTABLE_JOB_FIELDS.intersection(updates or {})
    if bad_fields:
        raise ValueError(
            f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}"
        )

    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue

            updated = {**job, **updates}

            # 如果更新了 schedule，重新计算 next_run_at
            if "schedule" in updates:
                schedule = updated["schedule"]
                if isinstance(schedule, str):
                    schedule = parse_schedule(schedule)
                    updated["schedule"] = schedule
                updated["schedule_display"] = updates.get(
                    "schedule_display",
                    schedule.get("display", updated.get("schedule_display")),
                )
                if updated.get("state") != STATE_PAUSED:
                    updated["next_run_at"] = compute_next_run(schedule)

            # 如果更新了 skills，规范化
            if "skills" in updates or "skill" in updates:
                normalized_skills = _normalize_skill_list(
                    updated.get("skill"), updated.get("skills")
                )
                updated["skills"] = normalized_skills
                updated["skill"] = normalized_skills[0] if normalized_skills else None

            # 确保暂停状态的作业不自动重新调度
            if updated.get("enabled", True) and updated.get("state") != STATE_PAUSED:
                if not updated.get("next_run_at"):
                    updated["next_run_at"] = compute_next_run(updated["schedule"])

            jobs[i] = updated
            save_jobs(jobs)
            return _normalize_job_record(jobs[i])

    return None


def pause_job(job_id: str, reason: str | None = None) -> dict[str, Any] | None:
    """暂停作业。"""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": False,
            "state": STATE_PAUSED,
            "paused_at": _now().isoformat(),
            "paused_reason": reason,
        },
    )


def resume_job(job_id: str) -> dict[str, Any] | None:
    """恢复暂停的作业。"""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    next_run_at = compute_next_run(job["schedule"])
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": STATE_SCHEDULED,
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": next_run_at,
        },
    )


def remove_job(job_id: str) -> bool:
    """删除作业及输出目录。"""
    job = resolve_job_ref(job_id)
    if not job:
        return False
    canonical_id = job["id"]
    with _jobs_lock():
        jobs = load_jobs()
        original_len = len(jobs)
        jobs = [j for j in jobs if j["id"] != canonical_id]
        if len(jobs) < original_len:
            job_output_dir = _job_output_dir(canonical_id)
            save_jobs(jobs)
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
            return True
    return False


def mark_job_run(
    job_id: str,
    success: bool,
    error: str | None = None,
) -> None:
    """标记作业已执行。

    更新 last_run_at、last_status、递增 completed 计数，
    计算 next_run_at，重复次数达上限时自动删除。
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue

            now = _now().isoformat()
            job["last_run_at"] = now
            job["last_status"] = "ok" if success else "error"
            job["last_error"] = error if not success else None
            job["fire_claim"] = None

            # 递增 completed 计数
            if job.get("repeat"):
                repeat = job["repeat"]
                completed = repeat.get("completed", 0) + 1
                repeat["completed"] = completed

                times = repeat.get("times")
                if times is not None and times > 0 and completed >= times:
                    jobs.pop(i)
                    save_jobs(jobs)
                    return

            # 计算下次执行
            job["next_run_at"] = compute_next_run(job["schedule"], now)

            if job["next_run_at"] is None:
                kind = job.get("schedule", {}).get("kind")
                if kind in ("cron", "interval"):
                    job["state"] = STATE_ERROR
                    if not job.get("last_error"):
                        job["last_error"] = "无法计算下次执行时间"
                else:
                    job["enabled"] = False
                    job["state"] = STATE_COMPLETED
            elif job.get("state") != STATE_PAUSED:
                job["state"] = STATE_SCHEDULED

            save_jobs(jobs)
            return


def save_job_output(job_id: str, output: str) -> Path | None:
    """保存作业输出到文件。"""
    ensure_dirs()
    job_output_dir = _job_output_dir(job_id)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(job_output_dir)

    timestamp = _now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = job_output_dir / f"{timestamp}.md"

    fd, tmp_path = tempfile.mkstemp(dir=str(job_output_dir), suffix=".tmp", prefix=".output_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_file)
        _secure_file(output_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return output_file


def claim_job_for_fire(
    job_id: str, *, claim_ttl_seconds: int = 300
) -> bool:
    """原子 CAS 抢占作业执行权。

    返回 True 表示当前调用者获得执行权，False 表示被其他进程抢占。
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] != job_id:
                continue
            if not job.get("enabled", True) or job.get("state") == STATE_PAUSED:
                return False
            now = _now()
            existing = job.get("fire_claim")
            if existing:
                try:
                    claimed_at = datetime.fromisoformat(existing["at"])
                    if (now - claimed_at).total_seconds() < claim_ttl_seconds:
                        return False
                except Exception:
                    pass
            job["fire_claim"] = {"at": now.isoformat(), "by": os.getpid()}
            kind = job.get("schedule", {}).get("kind")
            if kind in ("cron", "interval"):
                nxt = compute_next_run(job["schedule"], now.isoformat())
                if nxt:
                    job["next_run_at"] = nxt
            save_jobs(jobs)
            return True
        return False


def get_due_jobs() -> list[dict[str, Any]]:
    """获取所有到期可执行的作业。"""
    with _jobs_lock():
        return _get_due_jobs_locked()


def _get_due_jobs_locked() -> list[dict[str, Any]]:
    """_get_due_jobs 的内部实现，调用者必须持有 _jobs_lock。"""
    now = _now()
    raw_jobs = load_jobs()
    jobs = [copy.deepcopy(j) for j in raw_jobs]
    due: list[dict[str, Any]] = []

    for job in jobs:
        if not job.get("enabled", True):
            continue

        next_run = job.get("next_run_at")
        if not next_run:
            continue

        next_run_dt = datetime.fromisoformat(next_run)
        if next_run_dt.tzinfo is None:
            next_run_dt = next_run_dt.replace(tzinfo=now.tzinfo)

        if next_run_dt <= now:
            due.append(job)

    return due