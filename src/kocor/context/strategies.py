"""上下文策略。

定义 ContextStrategy 枚举和 ContextStrategyApplier 应用器。
"""

from __future__ import annotations

from kocor.config import Config
from kocor.context.budget import TokenBudget
from kocor.context.types import ContextStrategy, SummaryNode
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.llm_provider.message import Message


class ContextStrategyApplier:
    """上下文策略应用器。

    在每次 apply() 时根据策略类型选择合适的上下文管理方式处理消息列表。
    SLIDING_WINDOW 的 preserve_last/first_rounds 从 Config 读取。
    """

    def apply(
        self,
        messages: list[Message],
        strategy: ContextStrategy = ContextStrategy.DEFAULT,
        token_budget: TokenBudget | None = None,
    ) -> tuple[list[Message], SummaryNode | None]:
        """根据策略类型对消息列表应用上下文管理。

        Args:
            messages: 原始会话历史消息
            strategy: 上下文管理策略
            token_budget: Token 预算对象，提供 should_summarize/truncate 判断
                        用于在预算紧张时自动升级策略

        Returns:
            (处理后的消息列表, 摘要节点或 None)
        """
        if not messages:
            return [], None

        # Token 预算驱动的策略升级
        effective_strategy = strategy
        if token_budget is not None and token_budget.should_truncate():
            effective_strategy = ContextStrategy.AGGRESSIVE
        elif token_budget is not None and token_budget.should_summarize() and strategy == ContextStrategy.DEFAULT:
            effective_strategy = ContextStrategy.SLIDING_WINDOW

        if effective_strategy == ContextStrategy.DEFAULT:
            return messages, None

        if effective_strategy == ContextStrategy.AGGRESSIVE:
            preserve_last = 1
            preserve_first = 1
        elif effective_strategy == ContextStrategy.SLIDING_WINDOW:
            preserve_last = Config.get("preserve_last_rounds")
            preserve_first = Config.get("preserve_first_rounds")
        else:
            return messages, None

        window = SlidingWindowStrategy(
            preserve_last_rounds=preserve_last,
            preserve_first_rounds=preserve_first,
        )
        return window.apply(messages)
