"""Kocor Agent 会话管理模块。

提供可选的会话持久化、重置策略和会话切换能力。
通过 SessionManager 外观类对外提供统一 API。

使用方式：
    from kocor.session import SessionManager, SessionStore, SessionEntry, SessionResetPolicy

    manager = SessionManager(store=SessionStore(db_path=".kocor/sessions/db.sqlite"))
    entry = manager.get_or_create_session()
    manager.update_session(entry.session_key, message_count_delta=1)
"""

from __future__ import annotations

from kocor.session.manager import SessionManager
from kocor.session.store import SessionDB, SessionStore
from kocor.session.types import SessionEntry, SessionResetPolicy

__all__ = [
    "SessionManager",
    "SessionStore",
    "SessionDB",
    "SessionEntry",
    "SessionResetPolicy",
]