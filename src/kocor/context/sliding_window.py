"""滑动窗口策略。

将消息列表按轮次分割，支持三段落结构：
  [保留最开始 N 轮] + [摘要中间轮次] + [保留最近 N 轮]
"""

from __future__ import annotations

from kocor.context.summarizer import HistorySummarizer
from kocor.context.types import SummaryNode
from kocor.event.event_manager import EventEmitter
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.message import Message


class SlidingWindowStrategy:
    """滑动窗口策略。

    将消息列表按语义轮次（user -> assistant[tool chain] -> assistant[text]）分割。
    支持三段落结构：
    - 保留最开始 N 轮完整消息
    - 中间轮次压缩为摘要
    - 保留最近 N 轮完整消息

    Attributes:
        preserve_last_rounds: 保留的最近完整轮次数
        preserve_first_rounds: 保留的最开始完整轮次数
    """

    def __init__(
        self,
        preserve_last_rounds: int,
        preserve_first_rounds: int,
        event_emitter: EventEmitter | None = None,
        hook_manager: HookManager | None = None,
    ):
        self.summarizer = HistorySummarizer(
            event_emitter=event_emitter,
            hook_manager=hook_manager,
        )
        self.preserve_last_rounds = preserve_last_rounds
        self.preserve_first_rounds = preserve_first_rounds

    def apply(
        self,
        messages: list[Message],
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

        Returns:
            (处理后的消息列表, 摘要节点或 None)
        """
        if not messages:
            return [], None

        # 分割轮次
        rounds = self._split_into_rounds(messages)

        # first + last 已覆盖全部轮次时，不截断
        total_rounds = len(rounds)
        if total_rounds <= self.preserve_last_rounds + self.preserve_first_rounds:
            return messages, None

        # 三段落切割
        first = rounds[:self.preserve_first_rounds] if self.preserve_first_rounds > 0 else []
        last = rounds[-self.preserve_last_rounds:]
        middle = rounds[self.preserve_first_rounds:-self.preserve_last_rounds] \
            if self.preserve_first_rounds > 0 else rounds[:-self.preserve_last_rounds]

        # 无中间段落：直接拼接首尾
        if not middle:
            result: list[Message] = []
            for r in first:
                result.extend(r)
            for r in last:
                result.extend(r)
            return result, None

        # 中间段落压缩为摘要
        to_summarize = [msg for r in middle for msg in r]
        summary_node = self.summarizer.summarize(
            to_summarize,
            start_index=0,
            end_index=len(to_summarize),
        )
        summary_msg = Message(
            role="assistant",
            content=f"[历史对话摘要]\n{summary_node.summary}",
        )

        # 组装结果: first → summary → last
        result = []
        for r in first:
            result.extend(r)
        result.append(summary_msg)
        for r in last:
            result.extend(r)

        return result, summary_node

    def _split_into_rounds(self, messages: list[Message]) -> list[list[Message]]:
        """将消息列表分割为语义轮次。

        每次 LLM 调用产生一条 assistant 消息，以此为单位做滑动窗口；
        assistant 边界处切分（首次 assistant 不切）。
        连续 user 消息（中间无 assistant 回复，如 REPL 快速连续输入）
        视为同一回合的多条输入，合并到同一轮而不各自成轮，
        避免第一条 user 被孤立成单独一轮（BUG 3.5）。

        Args:
            messages: 消息列表

        Returns:
            轮次列表，每个轮次是一个消息列表
        """
        rounds: list[list[Message]] = []
        current_round: list[Message] = []

        for msg in messages:
            # assistant 消息始终开启新轮次；user 消息仅在跟随非 user
            # （即真正的对话回合切换）时才切分，连续 user 合并。
            if current_round and (
                msg.role == "assistant"
                or (msg.role == "user" and current_round[-1].role != "user")
            ):
                rounds.append(current_round)
                current_round = []

            current_round.append(msg)

        if current_round:
            rounds.append(current_round)

        return rounds
