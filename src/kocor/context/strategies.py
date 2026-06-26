"""上下文策略。

定义 ContextStrategy 枚举和 ContextStrategyApplier 应用器。
"""

from __future__ import annotations

from enum import Enum

from kocor.context.models import SummaryNode
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.context.summarizer import HistorySummarizer
from kocor.llm_provider.message import Message

# SLIDING_WINDOW 的默认保留轮次数
DEFAULT_PRESERVE_ROUNDS = 3


class ContextStrategy(Enum):
    """上下文管理策略。

    DEFAULT: 全量消息，无截断（适合短会话）
    SLIDING_WINDOW: 摘要旧轮次 + 保留最近 N 轮完整消息
    AGGRESSIVE: 仅保留最后一轮完整对话 + 其余历史摘要
    """

    DEFAULT = "default"
    SLIDING_WINDOW = "sliding"
    AGGRESSIVE = "aggressive"


class ContextStrategyApplier:
    """上下文策略应用器。

    持有 summarizer 和默认配置，在每次 apply() 时根据策略类型
    选择合适的上下文管理方式处理消息列表。

    Attributes:
        summarizer: 历史摘要器
        preserve_rounds: 保留的完整轮次数（None 则使用默认值 3）
    """

    def __init__(
        self,
        summarizer: HistorySummarizer,
        preserve_rounds: int | None = None,
    ):
        self.summarizer = summarizer
        self.preserve_rounds = preserve_rounds

    def apply(
        self,
        messages: list[Message],
        used_prompt: int,
        strategy: ContextStrategy = ContextStrategy.DEFAULT,
    ) -> tuple[list[Message], SummaryNode | None]:
        """根据策略类型对消息列表应用上下文管理。

        Args:
            messages: 原始会话历史消息
            used_prompt: 当前 prompt 已用 token
            strategy: 上下文管理策略

        Returns:
            (处理后的消息列表, 摘要节点或 None)
        """
        if not messages or strategy == ContextStrategy.DEFAULT:
            return messages, None

        if strategy == ContextStrategy.AGGRESSIVE:
            preserve = 1
        elif strategy == ContextStrategy.SLIDING_WINDOW:
            preserve = self.preserve_rounds or DEFAULT_PRESERVE_ROUNDS
        else:
            return messages, None

        window = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_rounds=preserve,
        )
        return window.apply(
            messages,
            current_usage=used_prompt,
        )
