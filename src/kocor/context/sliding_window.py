"""滑动窗口策略。

将消息列表按轮次分割，支持三段落结构：
  [保留最开始 N 轮] + [摘要中间轮次] + [保留最近 N 轮]
"""

from __future__ import annotations

from kocor.config import config_get
from kocor.context.types import SummaryNode
from kocor.context.summarizer import HistorySummarizer
from kocor.context.token_counter import TokenCounter
from kocor.llm_provider.message import Message


class SlidingWindowStrategy:
    """滑动窗口策略。

    将消息列表按语义轮次（user -> assistant[tool chain] -> assistant[text]）分割。
    支持三段落结构：
    - 保留最开始 N 轮完整消息
    - 中间轮次压缩为摘要
    - 保留最近 N 轮完整消息

    Attributes:
        summarizer: 历史摘要器
        preserve_last_rounds: 保留的最近完整轮次数
        preserve_first_rounds: 保留的最开始完整轮次数
        token_margin: token 余量（预留空间）
    """

    def __init__(
        self,
        summarizer: HistorySummarizer,
        preserve_last_rounds: int = 3,
        preserve_first_rounds: int = 1,
    ):
        self.summarizer = summarizer
        self.preserve_last_rounds = preserve_last_rounds
        self.preserve_first_rounds = preserve_first_rounds
        self._token_counter = TokenCounter()

    def apply(
        self,
        messages: list[Message],
        current_usage: int,
    ) -> tuple[list[Message], SummaryNode | None]:
        """对消息列表应用滑动窗口。

        三段落结构：
        1. 保留最开始 N 轮完整消息
        2. 中间轮次压缩为摘要（嵌入在返回消息列表的对应位置）
        3. 保留最近 N 轮完整消息

        当 preserve_first_rounds=0 时行为与原来一致。
        当 first + last >= 总轮次时不截断。

        Args:
            messages: 原始消息列表（不含 system prompt 和当前用户输入）
            current_usage: 当前已用 token（含 system prompt）

        Returns:
            (处理后的消息列表, 摘要节点或 None)
        """
        if not messages:
            return [], None

        # 分割轮次
        rounds = self._split_into_rounds(messages)

        # 估算 token 使用，检查是否需要紧急截断
        max_tokens = config_get("context_max_tokens", 200_000)
        token_margin = config_get("token_margin", 10_000)
        available = max_tokens - current_usage - token_margin
        history_tokens = self._token_counter.count_messages(messages)

        if history_tokens > available or available <= 0:
            if available <= 0 or history_tokens > max(available, 1) * 3:
                return self._aggressive_truncate(rounds)

        # 三段式逻辑
        total_rounds = len(rounds)
        if total_rounds <= self.preserve_last_rounds + self.preserve_first_rounds:
            # 轮次不足，不截断
            return messages, None

        # 分割三段
        first = rounds[:self.preserve_first_rounds] if self.preserve_first_rounds > 0 else []
        last = rounds[-self.preserve_last_rounds:]
        middle = rounds[self.preserve_first_rounds:-self.preserve_last_rounds] \
            if self.preserve_first_rounds > 0 else rounds[:-self.preserve_last_rounds]

        if not middle:
            # 中间无轮次，只需拼接首尾
            result: list[Message] = []
            for r in first:
                result.extend(r)
            for r in last:
                result.extend(r)
            return result, None

        # 中间轮次做摘要
        to_summarize = [msg for r in middle for msg in r]
        summary_node = self.summarizer.summarize(
            to_summarize,
            start_index=0,
            end_index=len(to_summarize),
        )
        summary_msg = Message(
            role="system",
            content=f"[历史对话摘要]\n{summary_node.summary}",
        )

        # 组装结果: first + summary + last
        result = []
        for r in first:
            result.extend(r)
        result.append(summary_msg)
        for r in last:
            result.extend(r)

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
                rounds.append(current_round)
                current_round = []

            current_round.append(msg)

        if current_round:
            rounds.append(current_round)

        return rounds

    def _aggressive_truncate(
        self, rounds: list[list[Message]],
    ) -> tuple[list[Message], SummaryNode | None]:
        """紧急截断：仅保留最后一轮，摘要嵌入返回结果。

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
        summary_msg = Message(
            role="system",
            content=f"[历史对话摘要]\n{summary_node.summary}",
        )
        result: list[Message] = [summary_msg]
        result.extend(last_round)
        return result, summary_node
