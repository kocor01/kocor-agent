"""上下文管理模块。

为 Agent 提供分层系统提示构建、Token 预算管理、记忆存储、
会话历史摘要与滑动窗口等上下文管理能力。
"""

from kocor.context.budget import TokenBudget
from kocor.context.memory import MemoryManager
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.summarizer import HistorySummarizer
from kocor.context.token_counter import TokenCounter
from kocor.context.context_manager import ContextManager
from kocor.context.types import (
    ContextStrategy,
    MemoryItem,
    SummaryNode,
)
from kocor.tools.truncate import ToolOutputTruncator

__all__ = [
    "ContextManager",
    "ContextStrategy",
    "ContextStrategyApplier",
    "HistorySummarizer",
    "MemoryItem",
    "MemoryManager",
    "SlidingWindowStrategy",
    "SummaryNode",
    "TokenBudget",
    "TokenCounter",
    "ToolOutputTruncator",
]
