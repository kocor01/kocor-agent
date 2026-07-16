"""MemoryStore 快照刷新测试。"""

from __future__ import annotations

import tempfile

from kocor.memory.store import MemoryStore
from kocor.memory.types import MemoryTarget


class TestMemorySnapshotFresh:
    def test_load_from_disk_creates_snapshot(self):
        store = MemoryStore(
            memory_dir=tempfile.mkdtemp(),
            memory_limit=5000,
            user_limit=2000,
            user_enabled=True,
        )
        snapshot = store.load_from_disk()
        assert snapshot.formatted_text is not None
        assert store.snapshot_version > 0

    def test_add_does_not_update_snapshot(self):
        """add 操作后快照不变（需手动 refresh_snapshot）。"""
        store = MemoryStore(
            memory_dir=tempfile.mkdtemp(),
            memory_limit=5000,
            user_limit=2000,
            user_enabled=True,
        )
        store.load_from_disk()
        old = store.format_for_system_prompt()
        v1 = store.snapshot_version

        store.add(MemoryTarget.MEMORY, "new memory entry")
        assert store.format_for_system_prompt() == old
        assert store.snapshot_version == v1

    def test_refresh_snapshot_updates_text(self):
        """refresh_snapshot 后快照反映最新磁盘内容。"""
        store = MemoryStore(
            memory_dir=tempfile.mkdtemp(),
            memory_limit=5000,
            user_limit=2000,
            user_enabled=True,
        )
        store.load_from_disk()
        old = store.format_for_system_prompt()

        store.add(MemoryTarget.MEMORY, "fresh memory")
        store.refresh_snapshot()

        new = store.format_for_system_prompt()
        assert new != old
        assert "fresh memory" in new

    def test_snapshot_version_increments_on_refresh(self):
        store = MemoryStore(
            memory_dir=tempfile.mkdtemp(),
            memory_limit=5000,
            user_limit=2000,
            user_enabled=True,
        )
        store.load_from_disk()
        v1 = store.snapshot_version
        store.refresh_snapshot()
        v2 = store.snapshot_version
        assert v2 > v1