"""持久记忆管理器。

负责记忆的 CRUD、文件持久化、索引维护。
使用文件系统存储，每条记忆一个 Markdown 文件 + YAML frontmatter。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from kocor.context.models import MemoryItem


class MemoryManager:
    """持久记忆管理器。

    每条记忆对应一个文件系统的 .md 文件，包含 YAML frontmatter。
    MEMORY.md 作为索引文件维护所有记忆的引用。

    Attributes:
        memory_dir: 记忆存储目录路径
    """

    def __init__(self, memory_dir: str | None = None):
        self.memory_dir = Path(memory_dir or self._default_dir())
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.memory_dir / "MEMORY.md"
        self._ensure_index()

    # ── 公共 API ────────────────────────────────────────

    def save(self, item: MemoryItem) -> None:
        """保存一条记忆（新建或更新）。

        Args:
            item: 要保存的记忆
        """
        now = datetime.now().isoformat(timespec="seconds")

        existing_file = self._find_by_name(item.name)
        if existing_file:
            # 更新已有文件
            file_path = self.memory_dir / existing_file
            item.updated_at = now
            if not item.created_at:
                item.created_at = now
        else:
            # 新建文件
            file_path = self.memory_dir / f"{item.name}.md"
            item.created_at = item.created_at or now
            item.updated_at = now

        frontmatter = self._build_frontmatter(item)
        file_path.write_text(f"{frontmatter}\n\n{item.content}")
        self._upsert_index(item.name, item.description)

    def get(self, name: str) -> MemoryItem | None:
        """按名称获取记忆。

        Args:
            name: 记忆名称

        Returns:
            MemoryItem 或 None（不存在时）
        """
        file_name = self._find_by_name(name)
        if not file_name:
            return None
        return self._read_file(self.memory_dir / file_name)

    def list(self) -> list[MemoryItem]:
        """列出所有持久记忆。

        Returns:
            记忆列表（空列表表示无记忆）
        """
        result = []
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            item = self._read_file(f)
            if item:
                result.append(item)
        return result

    def find_relevant(self, query: str, max_items: int = 5) -> list[MemoryItem]:
        """查找与查询相关的记忆。

        使用简单的关键词匹配 + 描述匹配，不引入向量搜索。

        Args:
            query: 查询字符串
            max_items: 最多返回条数

        Returns:
            按相关性排序的记忆列表
        """
        if not query or not query.strip():
            return []

        items = self.list()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[int, MemoryItem]] = []
        for item in items:
            score = 0
            searchable = (item.name + " " + item.description + " " + item.content).lower()

            # 精确匹配
            if query_lower in item.description.lower():
                score += 10
            if query_lower in item.name.lower():
                score += 5
            if query_lower in item.content.lower():
                score += 3

            # 词级别匹配
            content_words = set(searchable.split())
            common = query_words & content_words
            score += len(common) * 2

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:max_items]]

    def delete(self, name: str) -> bool:
        """删除一条记忆。

        Args:
            name: 记忆名称

        Returns:
            True 表示删除成功，False 表示记忆不存在
        """
        file_name = self._find_by_name(name)
        if not file_name:
            return False

        (self.memory_dir / file_name).unlink(missing_ok=True)
        self._remove_from_index(name)
        return True

    # ── 内部方法 ────────────────────────────────────────

    def _default_dir(self) -> str:
        """默认记忆目录。"""
        return str(Path.home() / ".kocor" / "memories")

    def _ensure_index(self) -> None:
        """确保 MEMORY.md 索引文件存在。"""
        if not self._index_path.exists():
            self._index_path.write_text(
                "# Kocor Agent 记忆索引\n\n"
                "每行: - [Title](file.md) — 描述\n",
            )

    def _build_frontmatter(self, item: MemoryItem) -> str:
        """构建 YAML frontmatter 字符串。"""
        body = f"""---
name: {item.name}
description: {item.description}
metadata:
  type: {item.memory_type}
created_at: {item.created_at}
updated_at: {item.updated_at}
---"""
        return body.strip()

    def _read_file(self, path: Path) -> MemoryItem | None:
        """从文件读取并解析为 MemoryItem。"""
        try:
            text = path.read_text()
        except Exception:
            return None

        if not text.startswith("---"):
            return None

        parts = text.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = parts[1]
        content = parts[2].strip()

        # 解析 frontmatter
        item = MemoryItem(
            name="",
            description="",
            content=content,
            memory_type="reference",
        )

        for line in frontmatter.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                item.name = line[len("name:"):].strip().strip("\"'")
            elif line.startswith("description:"):
                item.description = line[len("description:"):].strip().strip("\"'")
            elif line.startswith("type:"):
                item.memory_type = line[len("type:"):].strip().strip("\"'")
            elif line.startswith("created_at:"):
                item.created_at = line[len("created_at:"):].strip().strip("\"'")
            elif line.startswith("updated_at:"):
                item.updated_at = line[len("updated_at:"):].strip().strip("\"'")

        if not item.name:
            return None

        return item

    def _find_by_name(self, name: str) -> str | None:
        """在记忆目录中查找名称对应的文件名。"""
        # 精确查找 .md 文件
        candidate = self.memory_dir / f"{name}.md"
        if candidate.exists():
            return candidate.name
        return None

    def _upsert_index(self, name: str, description: str) -> None:
        """在 MEMORY.md 中插入或更新条目。"""
        if not self._index_path.exists():
            return

        lines = self._index_path.read_text("utf-8").splitlines()
        new_line = f"- [{name}]({name}.md) — {description}"
        found = False

        for i, line in enumerate(lines):
            if line.strip().startswith(f"- [{name}]"):
                lines[i] = new_line
                found = True
                break

        if not found:
            lines.append(new_line)

        self._index_path.write_text("\n".join(lines) + "\n")

    def _remove_from_index(self, name: str) -> None:
        """从 MEMORY.md 中移除条目。"""
        if not self._index_path.exists():
            return

        lines = self._index_path.read_text("utf-8").splitlines()
        lines = [l for l in lines if not l.strip().startswith(f"- [{name}]")]
        self._index_path.write_text("\n".join(lines) + "\n")