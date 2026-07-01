"""上下文管理数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kocor.context.budget import TokenBudget


class ContextStrategy(Enum):
    """上下文管理策略。

    DEFAULT: 全量消息，无截断（适合短会话）
    SLIDING_WINDOW: 保留最开始 N 轮 + 摘要中间轮次 + 保留最近 N 轮完整消息
    AGGRESSIVE: 仅保留最后一轮完整对话 + 其余历史摘要
    """

    DEFAULT = "default"
    SLIDING_WINDOW = "sliding"
    AGGRESSIVE = "aggressive"


@dataclass
class SummaryNode:
    """摘要节点，代表一段被压缩的历史。"""

    summary: str
    message_count: int
    token_count: int
    original_start: int
    original_end: int
