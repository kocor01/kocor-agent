"""测试 MemoryManager。"""

from __future__ import annotations

import os
import tempfile

from kocor.context.memory import MemoryManager
from kocor.context.models import MemoryItem


class TestMemoryManager:
    """测试 MemoryManager。"""

    def setup_method(self):
        """每个测试前创建临时目录。"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = MemoryManager(memory_dir=self.temp_dir)

    def _memory_path(self, name: str) -> str:
        return os.path.join(self.temp_dir, f"{name}.md")

    def _index_path(self) -> str:
        return os.path.join(self.temp_dir, "MEMORY.md")

    # ── 基本操作 ────────────────────────────────────────

    def test_init_creates_directory(self):
        """初始化时如果目录不存在应自动创建。"""
        new_dir = os.path.join(tempfile.mkdtemp(), "subdir", "memories")
        try:
            manager = MemoryManager(memory_dir=new_dir)
            assert os.path.isdir(new_dir)
        finally:
            import shutil
            shutil.rmtree(new_dir, ignore_errors=True)

    def test_init_creates_index(self):
        """初始化时应创建 MEMORY.md 索引文件。"""
        assert os.path.exists(self._index_path())

    def test_save_creates_file(self):
        """保存记忆应创建对应的 .md 文件。"""
        item = MemoryItem(
            name="test-item",
            description="测试项",
            content="测试内容",
            memory_type="reference",
        )
        self.manager.save(item)
        assert os.path.exists(self._memory_path("test-item"))

    def test_save_creates_file_with_frontmatter(self):
        """记忆文件应包含 YAML frontmatter。"""
        item = MemoryItem(
            name="test-item",
            description="测试项",
            content="测试内容正文",
            memory_type="user",
        )
        self.manager.save(item)
        content = open(self._memory_path("test-item"), encoding="utf-8").read()
        assert content.startswith("---")
        assert "name: test-item" in content
        assert "description: 测试项" in content
        assert "type: user" in content
        assert "测试内容正文" in content

    def test_save_updates_existing(self):
        """覆盖保存应更新已有文件。"""
        item1 = MemoryItem(name="test", description="原描述", content="原内容", memory_type="reference")
        self.manager.save(item1)

        item2 = MemoryItem(name="test", description="新描述", content="新内容", memory_type="user")
        self.manager.save(item2)

        retrieved = self.manager.get("test")
        assert retrieved is not None
        assert retrieved.description == "新描述"
        assert retrieved.content == "新内容"
        assert retrieved.memory_type == "user"

    def test_get_returns_item(self):
        """get() 应返回正确的 MemoryItem。"""
        item = MemoryItem(
            name="get-test",
            description="获取测试",
            content="获取测试内容",
            memory_type="reference",
        )
        self.manager.save(item)
        retrieved = self.manager.get("get-test")
        assert retrieved is not None
        assert retrieved.name == "get-test"
        assert retrieved.description == "获取测试"
        assert retrieved.content == "获取测试内容"

    def test_get_nonexistent_returns_none(self):
        """get() 不存在的记忆应返回 None。"""
        assert self.manager.get("nonexistent") is None

    def test_list_empty(self):
        """新建管理器应列出空列表。"""
        assert self.manager.list() == []

    def test_list_returns_all(self):
        """list() 应返回所有记忆。"""
        self.manager.save(MemoryItem(name="a", description="A", content="a", memory_type="reference"))
        self.manager.save(MemoryItem(name="b", description="B", content="b", memory_type="reference"))
        items = self.manager.list()
        assert len(items) == 2
        names = {i.name for i in items}
        assert names == {"a", "b"}

    def test_delete_removes_file(self):
        """delete() 应删除文件。"""
        self.manager.save(MemoryItem(name="del-me", description="x", content="x", memory_type="reference"))
        assert os.path.exists(self._memory_path("del-me"))
        result = self.manager.delete("del-me")
        assert result is True
        assert not os.path.exists(self._memory_path("del-me"))

    def test_delete_nonexistent_returns_false(self):
        """delete() 不存在的记忆应返回 False。"""
        assert self.manager.delete("nonexistent") is False

    def test_delete_removes_from_index(self):
        """delete() 应从索引中移除。"""
        self.manager.save(MemoryItem(name="remove-me", description="x", content="x", memory_type="reference"))
        index_before = open(self._index_path(), encoding="utf-8").read()
        assert "remove-me" in index_before

        self.manager.delete("remove-me")
        index_after = open(self._index_path(), encoding="utf-8").read()
        assert "remove-me" not in index_after

    # ── 边界情况 ────────────────────────────────────────

    def test_corrupted_file_skipped(self):
        """损坏的文件应被跳过（不崩溃）。"""
        with open(self._memory_path("corrupted"), "w", encoding="utf-8") as f:
            f.write("this is not valid frontmatter")
        items = self.manager.list()
        # 损坏文件被跳过，不在列表中
        names = [i.name for i in items]
        assert "corrupted" not in names

    def test_empty_content_saves(self):
        """空内容的记忆也能保存。"""
        item = MemoryItem(name="empty", description="空", content="", memory_type="reference")
        self.manager.save(item)
        retrieved = self.manager.get("empty")
        assert retrieved is not None
        assert retrieved.content == ""

    def test_name_with_spaces(self):
        """名称包含空格应正常处理。"""
        item = MemoryItem(name="my test item", description="带空格", content="test", memory_type="reference")
        self.manager.save(item)
        retrieved = self.manager.get("my test item")
        assert retrieved is not None
        assert retrieved.name == "my test item"

    # ── Index 维护 ───────────────────────────────────────

    def test_save_adds_to_index(self):
        """save() 应在 MEMORY.md 中添加条目。"""
        item = MemoryItem(name="index-test", description="索引测试", content="x", memory_type="reference")
        self.manager.save(item)
        index = open(self._index_path(), encoding="utf-8").read()
        assert "index-test" in index

    def test_index_format(self):
        """索引文件条目格式应正确。"""
        self.manager.save(
            MemoryItem(name="fmt-test", description="格式化测试", content="x", memory_type="reference"),
        )
        index = open(self._index_path(), encoding="utf-8").read()
        assert "- [fmt-test](fmt-test.md) — 格式化测试" in index

    # ── 检索 ────────────────────────────────────────────

    def test_find_relevant_empty_query(self):
        """空查询应返回空列表。"""
        self.manager.save(MemoryItem(name="a", description="A", content="a", memory_type="reference"))
        result = self.manager.find_relevant("")
        assert result == []

    def test_find_relevant_by_description(self):
        """应能通过描述匹配。"""
        self.manager.save(MemoryItem(
            name="python-skill", description="用户的 Python 技能", content="精通 Python", memory_type="user",
        ))
        self.manager.save(MemoryItem(
            name="js-skill", description="用户的 JavaScript 技能", content="熟悉 JS", memory_type="user",
        ))
        result = self.manager.find_relevant("Python")
        assert len(result) >= 1
        assert result[0].name == "python-skill"

    def test_find_relevant_by_content(self):
        """应能通过内容匹配。"""
        self.manager.save(MemoryItem(
            name="pref", description="偏好", content="用户喜欢简洁的回答", memory_type="feedback",
        ))
        result = self.manager.find_relevant("简洁")
        assert len(result) >= 1
        assert result[0].name == "pref"

    def test_find_relevant_max_items(self):
        """max_items 应限制返回数量。"""
        for i in range(10):
            self.manager.save(MemoryItem(
                name=f"item-{i}", description=f"测试项 {i}", content="keyword", memory_type="reference",
            ))
        result = self.manager.find_relevant("keyword", max_items=3)
        assert len(result) == 3

    def test_find_relevant_no_match(self):
        """无匹配时应返回空列表。"""
        self.manager.save(MemoryItem(name="a", description="A", content="a", memory_type="reference"))
        result = self.manager.find_relevant("zzz_not_exist_zzz")
        assert result == []
