"""上下文策略选择器。

根据配置的策略枚举，选择合适的上下文管理策略。
"""

from __future__ import annotations

from kocor.context.models import ContextStrategy, SummaryNode, TokenBudget
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.context.summarizer import HistorySummarizer
from kocor.llm_provider.message import Message

# SLIDING_WINDOW 的默认保留轮次数
DEFAULT_PRESERVE_ROUNDS = 3


def apply_context_strategy(
    messages: list[Message],
    token_budget: TokenBudget,
    summarizer: HistorySummarizer,
    strategy: ContextStrategy = ContextStrategy.DEFAULT,
    preserve_rounds: int | None = None,
) -> tuple[list[Message], SummaryNode | None]:
    """根据策略类型对消息列表应用上下文管理。

    Args:
        messages: 原始会话历史消息
        token_budget: 当前 token 预算
        summarizer: 历史摘要器
        strategy: 上下文管理策略
        preserve_rounds: 保留的完整轮次数（仅 SLIDING_WINDOW 和 AGGRESSIVE 有效）

    Returns:
        (处理后的消息列表, 摘要节点或 None)
    """
    if not messages or strategy == ContextStrategy.DEFAULT:
        return messages, None

    if strategy == ContextStrategy.AGGRESSIVE:
        window = SlidingWindowStrategy(
            summarizer=summarizer,
            preserve_rounds=1,
        )
        return window.apply(
            messages,
            max_tokens=token_budget.limit,
            current_usage=token_budget.used_prompt,
        )

    if strategy == ContextStrategy.SLIDING_WINDOW:
        window = SlidingWindowStrategy(
            summarizer=summarizer,
            preserve_rounds=preserve_rounds or DEFAULT_PRESERVE_ROUNDS,
        )
        return window.apply(
            messages,
            max_tokens=token_budget.limit,
            current_usage=token_budget.used_prompt,
        )

    return messages, None