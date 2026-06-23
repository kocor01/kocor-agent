"""滑动窗口策略。

将消息列表按轮次分割，保留最近的 N 轮完整消息，
将之前的轮次压缩为一段摘要。
"""

from __future__ import annotations

from kocor.context.models import SummaryNode
from kocor.context.summarizer import HistorySummarizer
from kocor.context.token_counter import TokenCounter
from kocor.llm_provider.message import Message


class SlidingWindowStrategy:
    """滑动窗口策略。

    将消息列表按语义轮次（user -> assistant[tool chain] -> assistant[text]）分割。
    保留最近的 N 轮完整消息，将之前的轮次压缩为一段摘要。

    Attributes:
        summarizer: 历史摘要器
        preserve_rounds: 保留的完整轮次数
        token_margin: token 余量（预留空间）
    """

    def __init__(
        self,
        summarizer: HistorySummarizer,
        preserve_rounds: int = 3,
        token_margin: int = 10_000,
    ):
        self.summarizer = summarizer
        self.preserve_rounds = preserve_rounds
        self.token_margin = token_margin
        self._token_counter = TokenCounter()

    def apply(
        self,
        messages: list[Message],
        max_tokens: int,
        current_usage: int,
    ) -> tuple[list[Message], SummaryNode | None]:
        """对消息列表应用滑动窗口。

        核心逻辑：按轮次分割 → 如果轮次超过 preserve_rounds → 旧轮次做摘要。
        Token 预算检查用于降级到紧急截断（即使轮次未超过限制）。

        Args:
            messages: 原始消息列表（不含 system prompt 和当前用户输入）
            max_tokens: 上下文窗口上限
            current_usage: 当前已用 token（含 system prompt）

        Returns:
            (处理后的消息列表, 摘要节点或 None)
        """
        if not messages:
            return [], None

        # 分割轮次
        rounds = self._split_into_rounds(messages)

        # 估算 token 使用，检查是否需要紧急截断
        available = max_tokens - current_usage - self.token_margin
        history_tokens = self._token_counter.count_messages(messages)

        if history_tokens > available or available <= 0:
            if available <= 0 or history_tokens > max(available, 1) * 3:
                # token 空间严重不足 → 紧急截断（仅保留最后一轮）
                return self._aggressive_truncate(rounds)

        if len(rounds) <= self.preserve_rounds:
            # 轮次少于保留数，不截断
            return messages, None

        # 保留最近的 N 轮，之前的做摘要
        preserve = rounds[-self.preserve_rounds:]
        summarize = rounds[:-self.preserve_rounds]

        result = [msg for round_msgs in preserve for msg in round_msgs]
        to_summarize = [msg for round_msgs in summarize for msg in round_msgs]

        summary_node = self.summarizer.summarize(
            to_summarize,
            start_index=0,
            end_index=len(to_summarize),
        )

        return result, summary_node

    def _split_into_rounds(self, messages: list[Message]) -> list[list[Message]]:
        """将消息列表分割为语义轮次。

        一轮以 user 消息开始，到下一个 user 消息或列表末尾结束。
        这对应 Agent 的一次迭代周期。

        Args:
            messages: 消息列表

        Returns:
            轮次列表，每个轮次是一个消息列表
        """
        rounds: list[list[Message]] = []
        current_round: list[Message] = []

        for msg in messages:
            if msg.role == "user" and current_round:
                # 上一个 user 消息开始新轮次
                rounds.append(current_round)
                current_round = []

            current_round.append(msg)

        if current_round:
            rounds.append(current_round)

        return rounds

    def _aggressive_truncate(
        self, rounds: list[list[Message]],
    ) -> tuple[list[Message], SummaryNode | None]:
        """紧急截断：仅保留最后一轮。

        Args:
            rounds: 轮次列表

        Returns:
            (保留的消息, 摘要节点)
        """
        if len(rounds) <= 1:
            return rounds[0] if rounds else [], None

        last_round = rounds[-1]
        earlier = [msg for r in rounds[:-1] for msg in r]

        summary_node = self.summarizer.summarize(earlier)
        return last_round, summary_node