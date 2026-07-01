"""测试会话存储层。

包括纯内存存储和 SQLite 持久化存储。使用临时文件测试 SQLite。
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from kocor.llm_provider.message import FunctionCall, Message, ToolCall
from kocor.session.store import SessionDB, SessionStore
from kocor.session.types import SessionEntry


@pytest.fixture
def entry() -> SessionEntry:
    return SessionEntry(
        session_key="kocor:default:cli",
        session_id="20260702_100000_a1b2c3d4",
        created_at=datetime(2026, 7, 2, 10, 0, 0),
        updated_at=datetime(2026, 7, 2, 10, 0, 0),
    )


class TestSessionStore:
    """纯内存 SessionStore 测试。"""

    def test_create_and_get(self, entry):
        store = SessionStore()
        store.set_entry(entry)
        assert store.get_entry("kocor:default:cli") is entry

    def test_get_nonexistent(self):
        store = SessionStore()
        assert store.get_entry("nonexistent") is None

    def test_delete_entry(self, entry):
        store = SessionStore()
        store.set_entry(entry)
        store.delete_entry("kocor:default:cli")
        assert store.get_entry("kocor:default:cli") is None

    def test_delete_nonexistent(self):
        store = SessionStore()
        store.delete_entry("nonexistent")  # 不应抛异常

    def test_update_entry(self, entry):
        store = SessionStore()
        store.set_entry(entry)
        entry.message_count = 10
        store.set_entry(entry)
        retrieved = store.get_entry("kocor:default:cli")
        assert retrieved is not None
        assert retrieved.message_count == 10

    def test_load_all_on_init(self):
        store = SessionStore()
        assert store.get_entry("anything") is None


class TestSessionDB:
    """SessionDB SQLite 持久化测试。"""

    @pytest.fixture
    def db_path(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            pass  # Windows 可能延迟释放文件锁

    @pytest.fixture
    def db(self, db_path) -> SessionDB:
        _db = SessionDB(db_path)
        yield _db
        _db.close()

    @pytest.fixture
    def sample_entry(self) -> SessionEntry:
        return SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=datetime(2026, 7, 2, 10, 0, 0),
            updated_at=datetime(2026, 7, 2, 10, 0, 0),
        )

    def test_create_and_load_entry(self, db, sample_entry):
        db.save_entry(sample_entry)
        loaded = db.load_entry("kocor:default:cli")
        assert loaded is not None
        assert loaded.session_key == sample_entry.session_key
        assert loaded.session_id == sample_entry.session_id
        assert loaded.created_at == sample_entry.created_at

    def test_load_nonexistent(self, db):
        assert db.load_entry("nonexistent") is None

    def test_load_all_entries(self, db):
        e1 = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=datetime(2026, 7, 2, 10, 0, 0),
            updated_at=datetime(2026, 7, 2, 10, 0, 0),
        )
        e2 = SessionEntry(
            session_key="kocor:work:cli",
            session_id="20260702_110000_e5f6g7h8",
            created_at=datetime(2026, 7, 2, 11, 0, 0),
            updated_at=datetime(2026, 7, 2, 11, 0, 0),
        )
        db.save_entry(e1)
        db.save_entry(e2)
        entries = db.load_all_entries()
        assert len(entries) == 2
        assert entries["kocor:default:cli"].session_id == "20260702_100000_a1b2c3d4"
        assert entries["kocor:work:cli"].session_id == "20260702_110000_e5f6g7h8"

    def test_update_entry(self, db, sample_entry):
        db.save_entry(sample_entry)
        sample_entry.message_count = 5
        sample_entry.total_tokens = 100
        db.save_entry(sample_entry)
        loaded = db.load_entry("kocor:default:cli")
        assert loaded is not None
        assert loaded.message_count == 5
        assert loaded.total_tokens == 100

    def test_end_and_reopen_session(self, db, sample_entry):
        db.save_entry(sample_entry)
        session_id = sample_entry.session_id

        # 结束会话
        db.end_session(session_id, "idle")
        session = db.get_session(session_id)
        assert session is not None
        assert session["end_reason"] == "idle"

        # 重新打开
        db.reopen_session(session_id)
        session = db.get_session(session_id)
        assert session is not None
        assert session["end_reason"] is None

    def test_session_count(self, db, sample_entry):
        assert db.session_count() == 0
        db.save_entry(sample_entry)
        assert db.session_count() == 1

    def test_append_and_get_messages(self, db, sample_entry):
        db.save_entry(sample_entry)
        session_id = sample_entry.session_id

        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi there!")
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'),
        )
        msg3 = Message(
            role="assistant",
            content="",
            tool_calls=[tool_call],
            reasoning="I need to read the file",
        )
        msg4 = Message(
            role="tool",
            content="file content",
            tool_call_id="call_1",
        )

        db.append_message(session_id, msg1)
        db.append_message(session_id, msg2)
        db.append_message(session_id, msg3)
        db.append_message(session_id, msg4)

        messages = db.get_messages(session_id)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"
        assert messages[2].role == "assistant"
        assert messages[2].tool_calls is not None
        assert len(messages[2].tool_calls) == 1
        assert messages[2].tool_calls[0].id == "call_1"
        assert messages[2].tool_calls[0].function.name == "read_file"
        assert messages[2].reasoning == "I need to read the file"
        assert messages[3].role == "tool"
        assert messages[3].content == "file content"
        assert messages[3].tool_call_id == "call_1"

    def test_append_message_with_token_count(self, db, sample_entry):
        db.save_entry(sample_entry)
        db.append_message(sample_entry.session_id, Message(role="user", content="Hello"), token_count=42)
        db.append_message(sample_entry.session_id, Message(role="assistant", content="Hi!"), token_count=7)

    def test_append_message_strips_leading_newlines(self, db, sample_entry):
        """content 开头多余的 \\n 应在持久化时被清洗。"""
        db.save_entry(sample_entry)
        db.append_message(
            sample_entry.session_id,
            Message(role="assistant", content="\n\n\n嗨~ 我是 Kocor"),
        )
        rows = db._conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", (sample_entry.session_id,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["content"] == "嗨~ 我是 Kocor"
        assert not rows[0]["content"].startswith("\n")

    def test_get_messages_empty(self, db, sample_entry):
        db.save_entry(sample_entry)
        messages = db.get_messages(sample_entry.session_id)
        assert messages == []

    def test_get_messages_nonexistent_session(self, db):
        messages = db.get_messages("nonexistent")
        assert messages == []

    def test_get_sessions_list(self, db):
        e1 = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=datetime(2026, 7, 2, 10, 0, 0),
            updated_at=datetime(2026, 7, 2, 10, 0, 0),
        )
        e2 = SessionEntry(
            session_key="kocor:work:cli",
            session_id="20260702_110000_e5f6g7h8",
            created_at=datetime(2026, 7, 2, 11, 0, 0),
            updated_at=datetime(2026, 7, 2, 11, 0, 0),
        )
        db.save_entry(e1)
        db.save_entry(e2)

        # 为第一个会话添加一条用户消息
        db.append_message(e1.session_id, Message(role="user", content="帮我分析这个文件"))

        sessions = db.get_sessions_list()
        assert len(sessions) == 2
        # 按创建时间降序
        assert sessions[0]["session_id"] == "20260702_110000_e5f6g7h8"
        assert sessions[1]["session_id"] == "20260702_100000_a1b2c3d4"
        # 第一条消息预览
        assert sessions[1]["title"] == "帮我分析这个文件"
        # 无消息的会话应返回空字符串
        assert sessions[0]["title"] == ""

    def test_session_id_exists(self, db, sample_entry):
        db.save_entry(sample_entry)
        assert db.session_id_exists(sample_entry.session_id) is True
        assert db.session_id_exists("nonexistent") is False

    def test_double_end_session_no_error(self, db, sample_entry):
        db.save_entry(sample_entry)
        db.end_session(sample_entry.session_id, "idle")
        db.end_session(sample_entry.session_id, "user_request")  # 再次结束不应抛异常
        session = db.get_session(sample_entry.session_id)
        assert session is not None
        assert session["end_reason"] == "user_request"  # 应被覆盖