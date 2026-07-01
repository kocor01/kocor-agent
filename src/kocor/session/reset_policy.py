"""会话重置策略评估。"""

from __future__ import annotations

from datetime import datetime, timedelta

from kocor.session.types import SessionEntry, SessionResetPolicy


def should_reset(
    entry: SessionEntry,
    policy: SessionResetPolicy,
    now: datetime | None = None,
) -> str | None:
    """评估会话是否需要重置。

    检查空闲超时（idle）和每日重置（daily）两个条件。

    Args:
        entry: 会话元数据
        policy: 重置策略配置
        now: 当前时间（用于测试注入）

    Returns:
        重置原因 "idle" 或 "daily"，无需重置时返回 None
    """
    if policy.mode == "none":
        return None

    now = now or datetime.now()

    if policy.mode in {"idle", "daily"}:
        # idle 检查
        if policy.mode == "idle":
            idle_deadline = entry.updated_at + timedelta(minutes=policy.idle_minutes)
            if now > idle_deadline:
                return "idle"

        # daily 检查
        if policy.mode == "daily":
            today_reset = now.replace(
                hour=policy.at_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
            if now.hour < policy.at_hour:
                today_reset -= timedelta(days=1)
            if entry.updated_at < today_reset:
                return "daily"

    return None
