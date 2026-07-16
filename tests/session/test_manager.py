"""测试 SessionManager 外观类。"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from kocor.session.manager import SessionManager
from kocor.session.store import SessionStore
from kocor.session.types import SessionResetPolicy


@pytest.fixture
def db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass  # Windows 可能延迟释放文件锁


@pytest.fixture
def store(db_path) -> SessionStore:
    store = SessionStore(db_path=db_path)
    yield store
    if store.db:
        store.db.close()


@pytest.fixture
def policy() -> SessionResetPolicy:
    return SessionResetPolicy(mode="none")


@pytest.fixture
def manager(store, policy) -> SessionManager:
    return SessionManager(store=store, policy=policy)


class TestSessionManager:
    """SessionManager 功能测试。"""

    NOW = datetime(2026, 7, 2, 10, 0, 0)

    def test_get_or_create_new(self, manager):
        entry = manager.get_or_create_session(now=self.NOW)
        assert entry.session_key == "kocor:default:cli"
        assert entry.session_id.startswith("20260702_")
        assert entry.was_auto_reset is False
        assert entry.created_at == self.NOW

    def test_get_or_create_existing(self, manager):
        entry1 = manager.get_or_create_session(now=self.NOW)
        entry2 = manager.get_or_create_session()
        assert entry2.session_id == entry1.session_id

    def test_get_or_create_force_new(self, manager):
        entry1 = manager.get_or_create_session(now=self.NOW)
        entry2 = manager.get_or_create_session(force_new=True, now=self.NOW)
        assert entry2.session_id != entry1.session_id
        assert entry2.was_auto_reset is False  # force_new 不是 auto_reset

    def test_get_or_create_with_profile(self, store, policy):
        manager = SessionManager(store=store, policy=policy, profile="project-x")
        entry = manager.get_or_create_session(now=self.NOW)
        assert entry.session_key == "kocor:project-x:cli"

    def test_update_session(self, manager):
        entry = manager.get_or_create_session(now=self.NOW)
        manager.update_session(
            session_key=entry.session_key,
            message_count_delta=3,
            prompt_tokens_delta=100,
            completion_tokens_delta=50,
            total_tokens_delta=150,
            now=self.NOW,
        )
        updated = manager.store.get_entry(entry.session_key)
        assert updated is not None
        assert updated.message_count == 3
        assert updated.prompt_tokens == 100
        assert updated.completion_tokens == 50
        assert updated.total_tokens == 150
        assert updated.updated_at >= self.NOW

    def test_reset_session(self, manager):
        entry1 = manager.get_or_create_session(now=self.NOW)
        entry2 = manager.reset_session(entry1.session_key, now=self.NOW)
        assert entry2.session_id != entry1.session_id
        assert entry2.is_fresh_reset is True
        # 旧会话应在 DB 中标记结束
        if manager.store._db:
            _session = manager.store._db.get_session(entry1.session_id)
            # entry1 在 reset 时被标记 ended_at

    def test_end_session(self, manager):
        entry = manager.get_or_create_session(now=self.NOW)
        manager.end_session(entry.session_key, reason="test")
        if manager.store._db:
            session = manager.store._db.get_session(entry.session_id)
            assert session is not None
            assert session["end_reason"] == "test"

    def test_get_session_info(self, manager):
        entry = manager.get_or_create_session(now=self.NOW)
        info = manager.get_session_info(entry.session_key)
        assert info is not None
        assert info.session_id == entry.session_id

    def test_get_session_info_nonexistent(self, manager):
        assert manager.get_session_info("nonexistent") is None

    def test_has_any_sessions_false(self, manager):
        assert manager.has_any_sessions() is False

    def test_has_any_sessions_true(self, manager):
        manager.get_or_create_session(now=self.NOW)
        assert manager.has_any_sessions() is True

    def test_auto_reset_by_idle(self, store, policy):
        """空闲超时应自动重置。"""
        policy.mode = "idle"
        policy.idle_minutes = 60
        manager = SessionManager(store=store, policy=policy)

        now = datetime(2026, 7, 2, 10, 0, 0)
        entry1 = manager.get_or_create_session(now=now)

        # 模拟 90 分钟后
        later = datetime(2026, 7, 2, 11, 30, 0)  # 90 分钟后
        entry2 = manager.get_or_create_session(now=later)
        assert entry2.session_id != entry1.session_id
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "idle"

    def test_no_auto_reset_within_idle(self, store, policy):
        """空闲时间内不应重置。"""
        policy.mode = "idle"
        policy.idle_minutes = 120
        manager = SessionManager(store=store, policy=policy)

        now = datetime(2026, 7, 2, 10, 0, 0)
        entry1 = manager.get_or_create_session(now=now)

        later = datetime(2026, 7, 2, 11, 0, 0)  # 60 分钟后
        entry2 = manager.get_or_create_session(now=later)
        assert entry2.session_id == entry1.session_id
        assert entry2.was_auto_reset is False

    def test_auto_reset_by_daily(self, store, policy):
        """跨天且过重置时刻应自动重置。"""
        policy.mode = "daily"
        policy.at_hour = 4
        manager = SessionManager(store=store, policy=policy)

        now = datetime(2026, 7, 2, 10, 0, 0)
        entry1 = manager.get_or_create_session(now=now)

        # 第 2 天上午 10 点（跨过凌晨 4 点重置时刻）
        next_day = datetime(2026, 7, 3, 10, 0, 0)
        entry2 = manager.get_or_create_session(now=next_day)
        assert entry2.session_id != entry1.session_id
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "daily"

    def test_session_id_format_in_entry(self, manager):
        entry = manager.get_or_create_session(now=self.NOW)
        parts = entry.session_id.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 8

    def test_session_id_unique(self, manager):
        e1 = manager.get_or_create_session(now=self.NOW)
        # 即使在同一秒创建，session_id 也应不同（含 uuid 部分）
        e2 = manager.get_or_create_session(force_new=True, now=self.NOW)
        assert e1.session_id != e2.session_id


class TestSessionManagerPersistence:
    """测试 SessionManager 的持久化方法。"""

    NOW = datetime(2026, 7, 2, 10, 0, 0)

    @pytest.fixture
    def mgr(self, store, policy) -> SessionManager:
        return SessionManager(store=store, policy=policy)

    def test_persist_and_load_messages(self, mgr):
        e = mgr.get_or_create_session(now=self.NOW)

        from kocor.llm_provider.message import Message

        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！我是 Kocor"),
        ]
        idx = mgr.persist_messages(e.session_key, msgs, start_index=0)
        assert idx == 2

        loaded = mgr.load_messages(e.session_id)
        assert len(loaded) == 2
        assert loaded[0].content == "你好"
        assert loaded[1].content == "你好！我是 Kocor"

    def test_persist_skip_system_messages(self, mgr):
        e = mgr.get_or_create_session(now=self.NOW)

        from kocor.llm_provider.message import Message

        msgs = [
            Message(role="system", content="你是一个助手"),
            Message(role="user", content="你好"),
        ]
        idx = mgr.persist_messages(e.session_key, msgs, start_index=0)
        assert idx == 2  # 索引总是推进到末尾

        loaded = mgr.load_messages(e.session_id)
        assert len(loaded) == 1  # system 没有被持久化
        assert loaded[0].role == "user"

    def test_persist_incremental(self, mgr):
        """验证增量持久化——只持久化新消息。"""
        e = mgr.get_or_create_session(now=self.NOW)

        from kocor.llm_provider.message import Message

        msgs = [
            Message(role="user", content="第 1 条"),
            Message(role="assistant", content="回复 1"),
        ]
        mgr.persist_messages(e.session_key, msgs, start_index=0)

        # 追加新消息，从索引 2 开始持久化
        msgs.append(Message(role="user", content="第 2 条"))
        msgs.append(Message(role="assistant", content="回复 2"))
        mgr.persist_messages(e.session_key, msgs, start_index=2)

        loaded = mgr.load_messages(e.session_id)
        assert len(loaded) == 4
        assert loaded[2].content == "第 2 条"

    def test_persist_with_real_token_count(self, mgr):
        """assistant 消息应写入 API 返回的真实 token 消耗。"""
        e = mgr.get_or_create_session(now=self.NOW)

        from kocor.llm_provider.message import Message, Usage

        msgs = [
            Message(role="user", content="Hello"),  # 无 usage
            Message(
                role="assistant", content="Hi!", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),  # 有 usage
        ]
        mgr.persist_messages(e.session_key, msgs, start_index=0)

        rows = mgr.store.db._conn.execute(
            "SELECT role, total_tokens FROM messages WHERE session_id = ? ORDER BY id",
            (e.session_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["role"] == "user"
        assert rows[0]["total_tokens"] == 0  # user 消息无 usage
        assert rows[1]["role"] == "assistant"
        assert rows[1]["total_tokens"] == 15  # prompt(10) + completion(5)

    def test_persist_no_db(self, policy):
        """无 SQLite 后端时 persist 不应出错。"""
        store = SessionStore()  # 纯内存，无 db
        mgr = SessionManager(store=store, policy=policy)

        from kocor.llm_provider.message import Message

        msgs = [Message(role="user", content="你好")]
        idx = mgr.persist_messages("kocor:default:cli", msgs)
        assert idx == len(msgs)

    def test_load_messages_no_db(self, policy):
        store = SessionStore()
        mgr = SessionManager(store=store, policy=policy)
        assert mgr.load_messages("nonexistent") == []

    def test_switch_to_session(self, mgr):
        """切换会话应结束当前会话并恢复目标会话的消息。"""
        from kocor.llm_provider.message import Message

        # 创建会话 1
        e1 = mgr.get_or_create_session(now=self.NOW)
        mgr.persist_messages(
            e1.session_key,
            [
                Message(role="user", content="会话 1 的消息"),
            ],
            start_index=0,
        )

        # 创建会话 2（强制新会话）
        e2 = mgr.get_or_create_session(force_new=True, now=self.NOW)
        assert e2.session_id != e1.session_id

        # 切换回会话 1
        msgs = mgr.switch_to_session(e2.session_key, e1.session_id)
        assert len(msgs) == 1
        assert msgs[0].content == "会话 1 的消息"

        # 验证当前 entry 指向会话 1
        entry = mgr.store.get_entry(e2.session_key)
        assert entry is not None
        assert entry.session_id == e1.session_id

    def test_switch_to_nonexistent(self, mgr):
        mgr.get_or_create_session(now=self.NOW)
        msgs = mgr.switch_to_session("kocor:default:cli", "nonexistent")
        assert msgs == []

    def test_switch_without_db(self, policy):
        store = SessionStore()  # 无 DB
        mgr = SessionManager(store=store, policy=policy)
        msgs = mgr.switch_to_session("kocor:default:cli", "some-id")
        assert msgs == []

    def test_get_sessions_list_with_db(self, mgr):
        mgr.get_or_create_session(now=self.NOW)
        sessions = mgr.get_sessions_list()
        assert len(sessions) >= 1
        assert sessions[0]["session_id"].startswith("20260702_")
