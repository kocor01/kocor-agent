"""SessionManager — 会话管理的外观（Facade）类。

对外提供统一的主要 API：获取/创建/更新/重置会话。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from kocor.session.reset_policy import should_reset
from kocor.session.session_key import build_session_key
from kocor.session.store import SessionStore
from kocor.session.types import SessionEntry, SessionResetPolicy
from kocor.llm_provider.message import Message


def _generate_session_id(now: datetime | None = None) -> str:
    """生成会话 ID。

    格式: ``YYYYMMDD_HHMMSS_<8hex>``
    """
    now = now or datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class SessionManager:
    """会话管理的外观类，对外提供主要 API。

    职责：
    - 统一的入口：获取/创建/更新/重置会话
    - 代理 SessionStore 和重置策略
    - 不关心消息内容，只维护会话元数据和状态

    Args:
        store: SessionStore 实例（可带 SQLite 后端）
        policy: 重置策略配置
        profile: 会话命名空间（用于 session_key 生成）
    """

    def __init__(
        self,
        store: SessionStore,
        policy: SessionResetPolicy | None = None,
        profile: str | None = None,
    ):
        self.store = store
        self.policy = policy or SessionResetPolicy()
        self.profile = profile

    @property
    def session_key(self) -> str:
        """当前会话键。"""
        return build_session_key(profile=self.profile)

    # -- 公开 API --

    def get_or_create_session(
        self,
        force_new: bool = False,
        now: datetime | None = None,
    ) -> SessionEntry:
        """获取或创建会话。

        进入点方法，包含重置策略评估逻辑。

        Args:
            force_new: 强制创建新会话（忽略已有会话）
            now: 当前时间（用于测试注入）

        Returns:
            当前的 SessionEntry
        """
        session_key = self.session_key
        now = now or datetime.now()

        existing = self.store.get_entry(session_key)

        if existing is not None and not force_new:
            reset_reason = should_reset(existing, self.policy, now=now)
            if reset_reason:
                # 自动重置
                self._end_old_session(existing.session_id, reset_reason)
                return self._create_entry(
                    session_key=session_key,
                    now=now,
                    was_auto_reset=True,
                    auto_reset_reason=reset_reason,
                )
            # 会话仍然有效，更新活跃时间
            existing.updated_at = now
            self.store.set_entry(existing)
            return existing

        # 创建新会话前结束旧会话（force_new 或首次创建）
        if existing is not None:
            self._end_old_session(existing.session_id, "force_new")

        # 创建新会话
        return self._create_entry(
            session_key=session_key,
            now=now,
        )

    def update_session(
        self,
        session_key: str,
        message_count_delta: int = 0,
        prompt_tokens_delta: int = 0,
        completion_tokens_delta: int = 0,
        total_tokens_delta: int = 0,
        cached_tokens_delta: int = 0,
        now: datetime | None = None,
    ) -> None:
        """更新会话元数据。

        Args:
            session_key: 要更新的会话键
            message_count_delta: 新增消息数
            prompt_tokens_delta: 新增输入 token 数
            completion_tokens_delta: 新增输出 token 数
            total_tokens_delta: 新增总 token 数
            cached_tokens_delta: 新增缓存 token 数
            now: 当前时间（用于测试注入）
        """
        entry = self.store.get_entry(session_key)
        if entry is None:
            return

        entry.updated_at = now or datetime.now()
        entry.message_count += message_count_delta
        entry.prompt_tokens += prompt_tokens_delta
        entry.completion_tokens += completion_tokens_delta
        entry.total_tokens += total_tokens_delta
        entry.cached_tokens += cached_tokens_delta

        self.store.set_entry(entry)

    def reset_session(
        self,
        session_key: str,
        now: datetime | None = None,
    ) -> SessionEntry:
        """显式重置会话（/reset 命令）。

        结束旧会话，创建新会话。新会话标记 ``is_fresh_reset=True``。

        Args:
            session_key: 要重置的会话键
            now: 当前时间（用于测试注入）

        Returns:
            新的 SessionEntry
        """
        now = now or datetime.now()
        existing = self.store.get_entry(session_key)
        if existing is not None:
            self._end_old_session(existing.session_id, "user_reset")

        return self._create_entry(
            session_key=session_key,
            now=now,
            is_fresh_reset=True,
        )

    def end_session(
        self,
        session_key: str,
        reason: str = "user_request",
    ) -> None:
        """结束当前会话（标记结束，保留历史）。

        Args:
            session_key: 要结束的会话键
            reason: 结束原因
        """
        entry = self.store.get_entry(session_key)
        if entry is not None and self.store.db:
            self.store.db.end_session(entry.session_id, reason)

    def get_session_info(self, session_key: str) -> SessionEntry | None:
        """查询会话元数据。"""
        return self.store.get_entry(session_key)

    def has_any_sessions(self) -> bool:
        """检查是否存在历史会话记录。"""
        return self.store.has_any()

    def get_sessions_list(self) -> list[dict[str, Any]]:
        """获取会话列表（用于 /sessions 命令）。

        需要 SQLite 后端支持。无后端时返回空列表。
        """
        if self.store.db:
            return self.store.db.get_sessions_list()
        return []

    def switch_to_session(
        self,
        session_key: str,
        target_session_id: str,
    ) -> list[Message]:
        """切换到指定会话（/session <id> 命令）。

        结束当前会话，重新打开目标会话（清除 end_reason），
        将该 session_key 映射指向目标 session_id，并返回其消息历史。

        Args:
            session_key: 当前会话键
            target_session_id: 目标会话 ID

        Returns:
            目标会话的消息历史（按时间升序）；无 DB 时返回空列表
        """
        if not self.store.db:
            return []

        # 结束当前会话
        current = self.store.get_entry(session_key)
        if current is not None and current.session_id != target_session_id:
            self.store.db.end_session(current.session_id, "switched")

        # 加载目标会话的消息
        messages = self.store.db.get_messages(target_session_id)

        # 重建目标会话的 entry（session 表里的行可能已被覆盖，所以从 messages 推算）
        now = datetime.now()
        target_row = self.store.db.get_session(target_session_id)
        if target_row is not None:
            self.store.db.reopen_session(target_session_id)
            created_at = datetime.fromisoformat(target_row["created_at"])
            msg_count = target_row["message_count"]
        else:
            # session 表行已被覆盖（如 force_new + replace），从 messages 推算
            created_at = now
            msg_count = len(messages)

        entry = SessionEntry(
            session_key=session_key,
            session_id=target_session_id,
            created_at=created_at,
            updated_at=now,
            message_count=msg_count,
        )
        self.store.set_entry(entry)  # 更新内存 dict + DB

        return messages

    def load_messages(self, session_id: str) -> list[Message]:
        """加载指定会话的消息历史。

        Args:
            session_id: 会话 ID

        Returns:
            消息列表（按时间升序）
        """
        if self.store.db:
            return self.store.db.get_messages(session_id)
        return []

    def persist_messages(
        self,
        session_key: str,
        messages: list[Message],
        start_index: int = 0,
    ) -> int:
        """持久化消息列表中的新增消息到 SQLite。

        从 start_index 开始遍历，将尚未持久化的消息写入 DB。
        跳过 system 角色消息。

        Args:
            session_key: 会话键
            messages: 全部消息列表
            start_index: 上次已持久化的消息索引（从 0 开始）

        Returns:
            已持久化的消息数（可用于下一轮的 start_index）
        """
        if not self.store.db:
            return len(messages)

        entry = self.store.get_entry(session_key)
        if entry is None:
            return len(messages)

        for i in range(start_index, len(messages)):
            msg = messages[i]
            if msg.role == "system":
                continue

            # assistant 消息记录 completion_tokens（该消息生成的 token 数），input 只在会话级统计
            self.store.db.append_message(entry.session_id, msg, usage=msg.usage)

        return len(messages)

    # -- 内部辅助 --

    def _create_entry(
        self,
        session_key: str,
        now: datetime,
        was_auto_reset: bool = False,
        auto_reset_reason: str | None = None,
        is_fresh_reset: bool = False,
    ) -> SessionEntry:
        session_id = _generate_session_id(now)
        entry = SessionEntry(
            session_key=session_key,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            was_auto_reset=was_auto_reset,
            auto_reset_reason=auto_reset_reason,
            is_fresh_reset=is_fresh_reset,
        )
        self.store.set_entry(entry)
        return entry

    def _end_old_session(self, session_id: str, reason: str) -> None:
        if self.store.db:
            self.store.db.end_session(session_id, reason)