"""上下文策略。

定义 ContextStrategy 枚举和 ContextStrategyApplier 应用器。
"""

from __future__ import annotations

from kocor.context.budget import TokenBudget
from kocor.context.types import ContextStrategy, SummaryNode
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.context.summarizer import HistorySummarizer
from kocor.llm_provider.message import Message

# SLIDING_WINDOW 的默认保留轮次数
DEFAULT_PRESERVE_LAST_ROUNDS = 3


class ContextStrategyApplier:
    """上下文策略应用器。

    持有 summarizer 和默认配置，在每次 apply() 时根据策略类型
    选择合适的上下文管理方式处理消息列表。

    Attributes:
        summarizer: 历史摘要器
        preserve_last_rounds: 保留的最近完整轮次数（None 则使用默认值 3）
        preserve_first_rounds: 保留的最开始完整轮次数（0 表示不保留）
    """

    def __init__(
        self,
        summarizer: HistorySummarizer,
        preserve_last_rounds: int | None = None,
        preserve_first_rounds: int = 1,
    ):
        self.summarizer = summarizer
        self.preserve_last_rounds = preserve_last_rounds
        self.preserve_first_rounds = preserve_first_rounds

    def apply(
        self,
        messages: list[Message],
        used_prompt: int,
        strategy: ContextStrategy = ContextStrategy.DEFAULT,
        token_budget: TokenBudget | None = None,
    ) -> tuple[list[Message], SummaryNode | None]:
        """根据策略类型对消息列表应用上下文管理。

        Args:
            messages: 原始会话历史消息
            used_prompt: 当前 prompt 已用 token（不含历史消息）
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
            preserve = 1
        elif effective_strategy == ContextStrategy.SLIDING_WINDOW:
            preserve = self.preserve_last_rounds or DEFAULT_PRESERVE_LAST_ROUNDS
        else:
            return messages, None

        window = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=preserve,
            preserve_first_rounds=self.preserve_first_rounds,
        )
        return window.apply(
            messages,
            current_usage=used_prompt,
        )
