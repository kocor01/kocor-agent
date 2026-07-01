"""记忆模块类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MemoryTarget(str, Enum):
    """记忆存储目标。"""

    MEMORY = "memory"  # Agent 个人笔记
    USER = "user"      # 用户画像


@dataclass
class MemorySnapshot:
    """会话启动时生成的冻结快照。

    快照在 load_from_disk() 时生成一次，会话内不变（保前缀缓存命中）。
    """

    memory_entries: list[str] = field(default_factory=list)
    user_entries: list[str] = field(default_factory=list)
    memory_usage: tuple[int, int] = (0, 0)   # (used, limit)
    user_usage: tuple[int, int] = (0, 0)
    formatted_text: str = ""                    # 注入 system prompt 的最终文本
