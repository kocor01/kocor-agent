"""SessionDB 批量提交测试。

通过 _pending_count 验证批量提交行为，避免 mock sqlite3 C 级方法。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from kocor.llm_provider.message import Message
from kocor.session.store import SessionDB


def _make_db(batch_mode=True, batch_threshold=20):
    """创建测试用的 SessionDB。"""
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    return SessionDB(str(db_path), batch_mode=batch_mode, batch_threshold=batch_threshold)


class TestBatchCommit:
    def test_batch_commit_reduces_commit_count(self):
        """批量模式下未达阈值时 pending 计数递增，flush 后落盘。"""
        db = _make_db(batch_mode=True, batch_threshold=5)

        for i in range(4):
            db.append_message("s1", Message(role="user", content=f"msg{i}"))

        # 4 < 阈值 5，pending 应为 4
        assert db._pending_count == 4, f"预期 pending=4, 实际={db._pending_count}"

        db.flush()
        assert db._pending_count == 0, "flush 后 pending 归零"

        assert len(db.get_messages("s1")) == 4

    def test_batch_commit_threshold(self):
        """达到阈值时自动提交（pending 归零）。"""
        db = _make_db(batch_mode=True, batch_threshold=3)

        for i in range(4):
            db.append_message("s1", Message(role="user", content=f"msg{i}"))

        # 第 3 条达到阈值触发提交，第 4 条后 pending=1
        assert db._pending_count == 1, f"预期 pending=1, 实际={db._pending_count}"

    def test_non_batch_mode_commits_immediately(self):
        """非批量模式下 pending 始终为 0。"""
        db = _make_db(batch_mode=False)

        for i in range(5):
            db.append_message("s1", Message(role="user", content=f"msg{i}"))

        assert db._pending_count == 0, "非批量模式 pending 应为 0"