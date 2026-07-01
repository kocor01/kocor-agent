"""测试记忆模块类型定义。"""

from __future__ import annotations

from kocor.memory.types import MemorySnapshot, MemoryTarget


class TestMemoryTarget:
    """测试 MemoryTarget 枚举。"""

    def test_enum_values(self):
        assert MemoryTarget.MEMORY.value == "memory"
        assert MemoryTarget.USER.value == "user"

    def test_two_targets(self):
        assert len(MemoryTarget) == 2


class TestMemorySnapshot:
    """测试 MemorySnapshot 数据模型。"""

    def test_default_values(self):
        snap = MemorySnapshot()
        assert snap.memory_entries == []
        assert snap.user_entries == []
        assert snap.memory_usage == (0, 0)
        assert snap.user_usage == (0, 0)
        assert snap.formatted_text == ""

    def test_with_entries(self):
        snap = MemorySnapshot(
            memory_entries=["fact one", "fact two"],
            memory_usage=(50, 2200),
            user_entries=["user named Alice"],
            user_usage=(30, 1375),
            formatted_text="## MEMORY (your personal notes)",
        )
        assert len(snap.memory_entries) == 2
        assert snap.memory_usage == (50, 2200)
        assert snap.formatted_text.startswith("## MEMORY")
