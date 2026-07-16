"""会话管理数据模型。

包含 SessionEntry（会话元数据）和 SessionResetPolicy（重置策略配置）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class SessionResetPolicy:
    """会话重置策略配置。

    控制会话何时自动失去上下文并开启新会话。

    Attributes:
        mode: 重置模式
            - "none" — 永不自动重置
            - "idle" — 空闲 N 分钟后重置
            - "daily" — 每天特定时刻重置
        idle_minutes: 空闲超时分钟数（仅 mode="idle" 时生效）
        at_hour: 每日重置时刻（0-23，仅 mode="daily" 时生效）
    """

    def __init__(
        self,
        mode: str = "none",
        idle_minutes: int = 1440,
        at_hour: int = 4,
    ):
        if mode not in ("none", "idle", "daily"):
            raise ValueError(f"无效的重置模式: '{mode}'，可选值: none, idle, daily")
        self.mode = mode
        self.idle_minutes = idle_minutes
        self.at_hour = at_hour


class SessionEntry:
    """会话元数据。

    代表一次会话的完整元数据信息。

    Attributes:
        session_key: 确定性会话键，如 "kocor:default:cli"
        session_id: 唯一实例 ID，格式 "YYYYMMDD_HHMMSS_<8hex>"
        created_at: 会话创建时间
        updated_at: 最后活动时间（用于空闲超时判定）
        title: 会话标题（从首个用户消息自动生成）
        message_count: 累计消息数（不含 system）
        prompt_tokens: 累计输入 token
        completion_tokens: 累计输出 token
        total_tokens: 累计总 token
        cached_tokens: 累计缓存 token
        was_auto_reset: 是否因策略自动重置
        auto_reset_reason: 自动重置原因（"idle" / "daily"）
        is_fresh_reset: 是否显式 /reset
    """

    def __init__(
        self,
        session_key: str,
        session_id: str,
        created_at: datetime,
        updated_at: datetime,
        title: str = "",
        message_count: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        was_auto_reset: bool = False,
        auto_reset_reason: str | None = None,
        is_fresh_reset: bool = False,
    ):
        self.session_key = session_key
        self.session_id = session_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.title = title
        self.message_count = message_count
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cached_tokens = cached_tokens
        self.was_auto_reset = was_auto_reset
        self.auto_reset_reason = auto_reset_reason
        self.is_fresh_reset = is_fresh_reset

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（用于 JSON / SQLite 存储）。"""
        return {
            "session_key": self.session_key,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "message_count": self.message_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "was_auto_reset": self.was_auto_reset,
            "auto_reset_reason": self.auto_reset_reason,
            "is_fresh_reset": self.is_fresh_reset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionEntry:
        """从 dict 反序列化。"""
        return cls(
            session_key=data["session_key"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            title=data.get("title", ""),
            message_count=data.get("message_count", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            was_auto_reset=data.get("was_auto_reset", False),
            auto_reset_reason=data.get("auto_reset_reason"),
            is_fresh_reset=data.get("is_fresh_reset", False),
        )

    def __repr__(self) -> str:
        return (
            f"SessionEntry(key={self.session_key}, "
            f"id={self.session_id}, "
            f"msgs={self.message_count})"
        )
