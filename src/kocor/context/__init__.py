"""上下文管理模块。

为 Agent 提供分层系统提示构建、Token 预算管理、记忆存储、
会话历史摘要与滑动窗口等上下文管理能力。
"""

from kocor.context.budget import TokenBudget
from kocor.context.builder import ContextBuilder
from kocor.context.memory import MemoryManager
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.summarizer import HistorySummarizer
from kocor.context.token_counter import TokenCounter
from kocor.context.session import AgentContext
from kocor.context.types import (
    ContextStrategy,
    MemoryItem,
    SummaryNode,
)
from kocor.tools.truncate import ToolOutputTruncator

__all__ = [
    "AgentContext",
    "ContextBuilder",
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
