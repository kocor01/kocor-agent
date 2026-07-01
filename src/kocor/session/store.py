"""会话存储层。

包含：
- SessionDB：SQLite 持久化层
- SessionStore：内存存储 + 可选 SQLite 后端
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from kocor.llm_provider.message import FunctionCall, Message, ToolCall, Usage
from kocor.session.types import SessionEntry


# ---------------------------------------------------------------------------
# SessionDB — SQLite 持久化层
# ---------------------------------------------------------------------------

class SessionDB:
    """SQLite 持久化层。

    WAL 模式，单线程访问。提供会话元数据和消息历史的原子化读写。
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id     TEXT PRIMARY KEY,
                session_key    TEXT NOT NULL,
                title          TEXT DEFAULT '',
                message_count  INTEGER DEFAULT 0,
                prompt_tokens  INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens   INTEGER DEFAULT 0,
                cached_tokens  INTEGER DEFAULT 0,
                was_auto_reset INTEGER DEFAULT 0,
                auto_reset_reason TEXT,
                is_fresh_reset INTEGER DEFAULT 0,
                ended_at       TEXT,
                end_reason     TEXT,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_session_key ON sessions(session_key);

            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES sessions(session_id),
                role         TEXT NOT NULL,
                content      TEXT NOT NULL DEFAULT '',
                tool_call_id TEXT,
                tool_name    TEXT,
                tool_calls   TEXT,
                reasoning    TEXT NOT NULL DEFAULT '',
                total_tokens INTEGER DEFAULT 0,
                prompt_tokens  INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                cached_tokens  INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
        """)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- 会话元数据 --

    def save_entry(self, entry: SessionEntry) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, session_key, title,
                    message_count, prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens,
                    was_auto_reset, auto_reset_reason, is_fresh_reset,
                    ended_at, end_reason, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.session_id,
                    entry.session_key,
                    entry.title,
                    entry.message_count,
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    entry.total_tokens,
                    entry.cached_tokens,
                    int(entry.was_auto_reset),
                    entry.auto_reset_reason,
                    int(entry.is_fresh_reset),
                    None,   # ended_at
                    None,   # end_reason
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                ),
            )
            self._conn.commit()

    def load_entry(self, session_key: str) -> SessionEntry | None:
        """加载指定 session_key 的最新活跃（无结束时间）会话。"""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_key = ? AND ended_at IS NULL ORDER BY updated_at DESC LIMIT 1",
            (session_key,),
        ).fetchone()
        if row is None:
            # 无活跃会话时，返回最近的一条（已完成的历史会话）
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_key = ? ORDER BY updated_at DESC LIMIT 1",
                (session_key,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def load_all_entries(self) -> dict[str, SessionEntry]:
        """加载每个 session_key 的最新非结束会话。"""
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY updated_at DESC"
        ).fetchall()
        result: dict[str, SessionEntry] = {}
        for row in rows:
            key = row["session_key"]
            if key not in result:
                result[key] = self._row_to_entry(row)
        return result

    def end_session(self, session_id: str, reason: str) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE session_id = ?",
            (now, reason, session_id),
        )
        self._conn.commit()

    def reopen_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def session_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()
        return row["cnt"] if row else 0

    def session_id_exists(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def update_session_title(self, session_id: str, title: str) -> None:
        """更新会话标题（取首个用户问题的摘要）。"""
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ? AND title = ''",
            (title[:50], session_id),
        )
        self._conn.commit()

    # -- 消息 --

    def append_message(
        self,
        session_id: str,
        message: Message,
        usage: Usage | None = None,
    ) -> None:
        # 持久化前清洗前导换行（模型常返回 "\n\n..." 格式的内容）
        content = message.content.lstrip("\n")
        tool_calls_json = None
        if message.tool_calls:
            tool_calls_json = json.dumps([
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ])

        tool_name = None
        if message.role == "tool" and message.tool_call_id:
            found = self._lookup_tool_call(session_id, message.tool_call_id)
            if found:
                tool_name = found["function"]["name"]
                tool_calls_json = json.dumps(found)

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cached_tokens = usage.cached_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        with self._lock:
            self._conn.execute(
                """INSERT INTO messages
                   (session_id, role, content, tool_call_id, tool_name,
                    tool_calls, reasoning, total_tokens,
                    prompt_tokens, completion_tokens, cached_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    message.role,
                    content,
                    message.tool_call_id,
                    tool_name,
                    tool_calls_json,
                    message.reasoning or "",
                    total_tokens,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    datetime.now().isoformat(),
                ),
            )
            self._conn.commit()

        # 自动以首个用户消息设置会话标题
        if message.role == "user" and content.strip():
            self.update_session_title(session_id, content.strip())

    def _lookup_tool_call(self, session_id: str, tool_call_id: str) -> dict | None:
        """查找与 tool_call_id 匹配的完整 tool_call 信息（用于写入 tool 消息的 tool_calls 列供审查）。"""
        row = self._conn.execute(
            """SELECT tool_calls FROM messages
               WHERE session_id = ? AND role = 'assistant'
                 AND tool_calls IS NOT NULL
               ORDER BY id DESC""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            calls = json.loads(row["tool_calls"])
            for tc in calls:
                if tc.get("id") == tool_call_id:
                    return tc
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def get_messages(self, session_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_sessions_list(self) -> list[dict]:
        """返回会话列表信息（用于 /sessions 命令）。

        每条记录包含 session_id, created_at, message_count, title。
        按 created_at 降序排列。
        """
        rows = self._conn.execute(
            """SELECT s.session_id, s.created_at, s.message_count, s.title
               FROM sessions s
               ORDER BY s.created_at DESC"""
        ).fetchall()
        result = []
        for row in rows:
            title = row["title"] or ""
            if len(title) > 30:
                title = title[:30] + "..."
            result.append({
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "message_count": row["message_count"],
                "title": title,
            })
        return result

    # -- 内部辅助 --

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> SessionEntry:
        return SessionEntry(
            session_key=row["session_key"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            title=row["title"],
            message_count=row["message_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            cached_tokens=row["cached_tokens"],
            was_auto_reset=bool(row["was_auto_reset"]),
            auto_reset_reason=row["auto_reset_reason"],
            is_fresh_reset=bool(row["is_fresh_reset"]),
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        tool_calls = None
        if row["tool_calls"]:
            try:
                raw = json.loads(row["tool_calls"])
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                        type=tc.get("type", "function"),
                    )
                    for tc in raw
                ]
            except (json.JSONDecodeError, KeyError, TypeError):
                tool_calls = None

        return Message(
            role=row["role"],
            content=row["content"] or "",
            tool_call_id=row["tool_call_id"],
            reasoning=row["reasoning"] or "",
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# SessionStore — 内存存储 + 可选 SQLite 后端
# ---------------------------------------------------------------------------

class SessionStore:
    """会话存储。

    提供基于 dict 的内存存储，可选 SQLite 后端支持持久化。

    当提供了 db_path 时，所有写操作同时写入 SQLite。
    读操作优先从内存返回。
    """

    def __init__(self, db_path: str | None = None):
        self._entries: dict[str, SessionEntry] = {}
        self._db: SessionDB | None = None
        if db_path:
            self._db = SessionDB(db_path)
            self._load_from_db()

    def _load_from_db(self) -> None:
        """从 SQLite 加载全部条目到内存。"""
        if self._db:
            self._entries = self._db.load_all_entries()

    def get_entry(self, session_key: str) -> SessionEntry | None:
        return self._entries.get(session_key)

    def set_entry(self, entry: SessionEntry) -> None:
        self._entries[entry.session_key] = entry
        if self._db:
            self._db.save_entry(entry)

    def delete_entry(self, session_key: str) -> None:
        self._entries.pop(session_key, None)

    def has_any(self) -> bool:
        return len(self._entries) > 0

    @property
    def db(self) -> SessionDB | None:
        return self._db
