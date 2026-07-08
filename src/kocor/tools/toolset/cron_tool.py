"""Cron 作业管理工具。

向 LLM 暴露单个压缩工具 cronjob(action=...)，通过 action 参数分发所有操作。
避免多个工具函数的 Schema 膨胀。

遵循 Hermes 的"单入口压缩工具"设计模式。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from kocor.tools.toolset.cron.jobs import (
    AmbiguousJobReference,
    claim_job_for_fire,
    create_job,
    get_job,
    list_jobs,
    mark_job_run,
    pause_job,
    remove_job,
    resolve_job_ref,
    resume_job,
    update_job,
)
from kocor.tools.toolset.cron.scanner import scan_cron_prompt

logger = logging.getLogger(__name__)

# =============================================================================
# 工具函数
# =============================================================================


def _format_job(job: dict[str, Any]) -> dict[str, Any]:
    """格式化作业为响应友好的结构。"""
    prompt = str(job.get("prompt") or "")
    skills = job.get("skills") or []
    job_id = str(job.get("id") or "unknown")
    name = str(
        job.get("name")
        or prompt[:50]
        or (skills[0] if skills else "")
        or job_id
        or "cron job"
    )
    return {
        "job_id": job_id,
        "name": name,
        "prompt": prompt,
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "skills": skills,
        "schedule": job.get("schedule_display") or "?",
        "repeat": _repeat_display(job),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "enabled": job.get("enabled", True),
        "state": job.get("state", "scheduled" if job.get("enabled", True) else "paused"),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
        "script": job.get("script"),
        "no_agent": job.get("no_agent", False),
    }


def _repeat_display(job: dict[str, Any]) -> str:
    times = (job.get("repeat") or {}).get("times")
    completed = (job.get("repeat") or {}).get("completed", 0)
    if times is None:
        return "forever"
    if times == 1:
        return "once" if completed == 0 else "1/1"
    return f"{completed}/{times}" if completed else f"{times} times"


def _canonical_skills(
    skill: str | None = None, skills: list[str] | None = None
) -> list[str]:
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


def _json_result(success: bool, **kwargs: Any) -> str:
    """构建 JSON 响应。"""
    result: dict[str, Any] = {"success": success}
    result.update(kwargs)
    if not success and "error" not in kwargs:
        result["error"] = "unknown error"
    return json.dumps(result, indent=2, ensure_ascii=False)


# =============================================================================
# 核心工具函数
# =============================================================================


def cronjob(
    action: str,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[str] = None,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    skill: Optional[str] = None,
    skills: Optional[list[str]] = None,
    script: Optional[str] = None,
    no_agent: bool = False,
    reason: Optional[str] = None,
    context_from: Optional[list[str]] = None,
    enabled_toolsets: Optional[list[str]] = None,
    workdir: Optional[str] = None,
) -> str:
    """统一的 cron 作业管理工具。

    Args:
        action: create | list | update | pause | resume | remove | run
        job_id: 作业 ID（update/pause/resume/remove/run 需要）
        prompt: 提示词（create/update 使用）
        schedule: 调度计划（create 必填）
        name: 可选名称
        repeat: 重复次数
        skill: 单技能名称
        skills: 技能列表
        script: 脚本路径（no_agent 模式）
        no_agent: True=仅执行脚本，跳过 LLM
        reason: 暂停原因
        context_from: 上游作业 ID 列表
        enabled_toolsets: 限制工具集
        workdir: 工作目录

    Returns:
        JSON 字符串
    """
    try:
        normalized = (action or "").strip().lower()

        # ---- create ----
        if normalized == "create":
            if not schedule:
                return _json_result(
                    False, error="schedule is required for create"
                )

            # prompt 或 skills 至少一个
            canonical_skills = _canonical_skills(skill, skills)
            _no_agent = bool(no_agent)
            prompt_text = prompt or ""

            if _no_agent:
                if not script:
                    return _json_result(
                        False,
                        error="create with no_agent=True requires a script — the script is the job.",
                    )
            elif not prompt_text and not canonical_skills:
                return _json_result(
                    False,
                    error="create requires either prompt or at least one skill",
                )

            # 安全扫描 prompt
            if prompt_text:
                scan_error = scan_cron_prompt(prompt_text)
                if scan_error:
                    return _json_result(False, error=scan_error)

            job = create_job(
                prompt=prompt_text or None,
                schedule=schedule,
                name=name,
                repeat=repeat,
                skill=skill,
                skills=canonical_skills or None,
                script=script,
                no_agent=_no_agent,
                context_from=context_from,
                enabled_toolsets=enabled_toolsets,
                workdir=workdir,
            )

            return _json_result(
                True,
                job_id=job["id"],
                name=job["name"],
                schedule=job["schedule_display"],
                repeat=_repeat_display(job),
                next_run_at=job["next_run_at"],
                job=_format_job(job),
                message=f"Cron job '{job['name']}' created.",
            )

        # ---- list ----
        if normalized == "list":
            jobs = [_format_job(job) for job in list_jobs(include_disabled=True)]
            return _json_result(True, count=len(jobs), jobs=jobs)

        # ---- 以下操作需要 job_id ----
        if not job_id:
            return _json_result(
                False, error=f"job_id is required for action '{normalized}'"
            )

        try:
            job = resolve_job_ref(job_id)
        except AmbiguousJobReference as exc:
            return _json_result(
                False,
                error=str(exc),
                matches=[
                    {
                        "id": m["id"],
                        "name": m.get("name"),
                        "schedule": m.get("schedule_display"),
                        "next_run_at": m.get("next_run_at"),
                    }
                    for m in exc.matches
                ],
            )

        if not job:
            return _json_result(
                False,
                error=f"Job with ID or name '{job_id}' not found. Use cronjob(action='list') to inspect jobs.",
            )

        # 解析为规范 ID
        job_id = job["id"]

        # ---- remove ----
        if normalized == "remove":
            removed = remove_job(job_id)
            if not removed:
                return _json_result(False, error=f"Failed to remove job '{job_id}'")
            return _json_result(
                True,
                message=f"Cron job '{job['name']}' removed.",
                removed_job={
                    "id": job_id,
                    "name": job["name"],
                    "schedule": job.get("schedule_display"),
                },
            )

        # ---- pause ----
        if normalized == "pause":
            updated = pause_job(job_id, reason=reason)
            if not updated:
                return _json_result(False, error=f"Failed to pause job '{job_id}'")
            return _json_result(True, job=_format_job(updated))

        # ---- resume ----
        if normalized == "resume":
            updated = resume_job(job_id)
            if not updated:
                return _json_result(False, error=f"Failed to resume job '{job_id}'")
            return _json_result(True, job=_format_job(updated))

        # ---- run / run_now / trigger ----
        if normalized in ("run", "run_now", "trigger"):
            # CAS 抢占执行权
            claimed = claim_job_for_fire(job_id)
            result_data = _format_job(get_job(job_id) or {"id": job_id})
            result_data["executed"] = claimed
            if claimed:
                # 记录执行（无 scheduler 时的简化处理）
                mark_job_run(job_id, True)
                refreshed = get_job(job_id) or {}
                result_data = _format_job(refreshed)
                result_data["executed"] = True
                result_data["execution_success"] = True
            else:
                result_data["execution_skipped"] = (
                    "Job is already being fired; not run again."
                )
            return _json_result(True, job=result_data)

        # ---- update ----
        if normalized == "update":
            updates: dict[str, Any] = {}

            if prompt is not None:
                scan_error = scan_cron_prompt(prompt)
                if scan_error:
                    return _json_result(False, error=scan_error)
                updates["prompt"] = prompt

            if name is not None:
                updates["name"] = name

            if skills is not None or skill is not None:
                canonical = _canonical_skills(skill, skills)
                updates["skills"] = canonical
                updates["skill"] = canonical[0] if canonical else None

            if schedule is not None:
                updates["schedule"] = schedule

            if repeat is not None:
                normalized_repeat = None if repeat <= 0 else repeat
                repeat_state = dict(job.get("repeat") or {})
                repeat_state["times"] = normalized_repeat
                updates["repeat"] = repeat_state

            if script is not None:
                updates["script"] = script or None

            if no_agent is not False:
                updates["no_agent"] = True

            if enabled_toolsets is not None:
                updates["enabled_toolsets"] = enabled_toolsets or None

            if workdir is not None:
                updates["workdir"] = workdir or None

            if not updates:
                return _json_result(False, error="No updates provided.")

            updated = update_job(job_id, updates)
            if not updated:
                return _json_result(False, error=f"Failed to update job '{job_id}'")
            return _json_result(True, job=_format_job(updated))

        # ---- unknown ----
        return _json_result(False, error=f"Unknown cron action '{action}'")

    except Exception as e:
        logger.exception("cronjob tool error")
        return _json_result(False, error=str(e))


# =============================================================================
# Schema 定义
# =============================================================================

CRONJOB_SCHEMA = {
    "name": "cronjob",
    "description": """管理定时任务（cron job）的单一工具。

action='create' 创建新定时任务。schedule 和 prompt 必填。
action='list' 查看所有任务。
action='update'、'pause'、'resume'、'remove' 或 'run' 管理已有任务。

注意：任务在独立的会话中执行，不携带当前对话上下文，prompt 必须自包含。
"""

    "",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "update", "pause", "resume", "remove", "run"],
                "description": "create: 创建 | list: 列表 | update: 更新 | pause: 暂停 | resume: 恢复 | remove: 删除 | run: 立即执行",
            },
            "job_id": {
                "type": "string",
                "description": "update/pause/resume/remove/run 时需要",
            },
            "prompt": {
                "type": "string",
                "description": "create/update 时使用，需自包含的提示词。如果同时指定了 skills，则作为任务指令与 skills 配合使用",
            },
            "schedule": {
                "type": "string",
                "description": "create 必填。'30m'（30分钟后）、'every 2h'（每2小时）、'0 9 * * *'（cron 表达式）、或 '2026-07-08T14:00:00'（ISO 时间戳）",
            },
            "name": {
                "type": "string",
                "description": "可选的人类可读名称",
            },
            "repeat": {
                "type": "integer",
                "description": "执行次数，省略则使用默认值（一次性=1，循环=无限），0 或负数=无限",
            },
            "skill": {
                "type": "string",
                "description": "单个技能名称（兼容旧版，建议使用 skills）",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "技能列表",
            },
            "script": {
                "type": "string",
                "description": "脚本路径（与 no_agent 配合使用）",
            },
            "no_agent": {
                "type": "boolean",
                "description": "True=仅执行脚本，跳过 LLM。要求 script 必填",
            },
            "reason": {
                "type": "string",
                "description": "暂停原因",
            },
            "context_from": {
                "type": "array",
                "items": {"type": "string"},
                "description": "上游作业 ID 列表，这些作业的最近输出会注入到 prompt 中",
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "限制 cron 作业可用的工具集列表",
            },
            "workdir": {
                "type": "string",
                "description": "工作目录绝对路径",
            },
        },
        "required": ["action"],
    },
}


# =============================================================================
# CronTool 类（遵循 Kocor 现有工具类模式）
# =============================================================================


class CronTool:
    """定时任务管理工具。"""

    NAME = "cronjob"
    DESCRIPTION = CRONJOB_SCHEMA["description"]
    SAFETY_LEVEL = "caution"
    PARAMETERS = CRONJOB_SCHEMA["parameters"]

    @staticmethod
    def handler(**kwargs: Any) -> str:
        """处理 cronjob 工具调用。"""
        return cronjob(**kwargs)