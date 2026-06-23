"""测试上下文策略选择器。"""

from __future__ import annotations

from kocor.context.models import ContextStrategy, TokenBudget
from kocor.context.strategies import apply_context_strategy
from kocor.context.summarizer import HistorySummarizer
from kocor.llm_provider.message import Message


class FakeLLM:
    def __init__(self):
        self.summary_text = "这是历史摘要"

    @property
    def provider(self):
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        return Message(role="assistant", content=self.summary_text)


class TestApplyContextStrategy:
    """测试 apply_context_strategy 策略选择器。"""

    def setup_method(self):
        self.summarizer = HistorySummarizer(llm=FakeLLM())

    def test_default_no_history(self):
        """DEFAULT 策略下空历史应原样返回。"""
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]
        budget = TokenBudget(limit=200_000, used_prompt=50)
        result, summary = apply_context_strategy(
            messages=msgs,
            token_budget=budget,
            summarizer=self.summarizer,
            strategy=ContextStrategy.DEFAULT,
        )
        assert summary is None
        assert result == msgs

    def test_default_with_long_history(self):
        """DEFAULT 策略下长历史也不截断。"""
        msgs = [Message(role="user", content=f"msg{i}") for i in range(100)]
        budget = TokenBudget(limit=200_000, used_prompt=50)
        result, summary = apply_context_strategy(
            messages=msgs,
            token_budget=budget,
            summarizer=self.summarizer,
            strategy=ContextStrategy.DEFAULT,
        )
        assert summary is None
        assert len(result) == 100

    def test_sliding_window_truncates(self):
        """SLIDING_WINDOW 策略应截断超出轮次的消息。"""
        from kocor.context.sliding_window import SlidingWindowStrategy
        msgs = []
        for i in range(10):
            msgs.extend([
                Message(role="user", content=f"问题{i}"),
                Message(role="assistant", content=f"回答{i}"),
            ])
        budget = TokenBudget(limit=200_000, used_prompt=50)
        result, summary = apply_context_strategy(
            messages=msgs,
            token_budget=budget,
            summarizer=self.summarizer,
            strategy=ContextStrategy.SLIDING_WINDOW,
            preserve_rounds=2,
        )
        assert summary is not None
        assert len(result) < len(msgs)

    def test_aggressive_preserves_one_round(self):
        """AGGRESSIVE 策略应仅保留最后一轮。"""
        msgs = []
        for i in range(5):
            msgs.extend([
                Message(role="user", content=f"问题{i}"),
                Message(role="assistant", content=f"回答{i}"),
            ])
        budget = TokenBudget(limit=200_000, used_prompt=50)
        result, summary = apply_context_strategy(
            messages=msgs,
            token_budget=budget,
            summarizer=self.summarizer,
            strategy=ContextStrategy.AGGRESSIVE,
        )
        assert summary is not None
        assert len(result) <= 3  # 最后 1 轮（2 条）+ margin

    def test_empty_messages(self):
        """空消息列表应返回空。"""
        budget = TokenBudget(limit=200_000, used_prompt=50)
        result, summary = apply_context_strategy(
            messages=[],
            token_budget=budget,
            summarizer=self.summarizer,
            strategy=ContextStrategy.SLIDING_WINDOW,
        )
        assert result == []
        assert summary is None

    def test_tight_budget_triggers_truncation(self):
        """token 预算紧张时 SLIDING_WINDOW 应触发截断。"""
        msgs = [Message(role="user", content=f"msg{i}") for i in range(30)]
        msgs.extend([Message(role="assistant", content=f"ans{i}") for i in range(30)])
        budget = TokenBudget(limit=500, used_prompt=400)
        result, summary = apply_context_strategy(
            messages=msgs,
            token_budget=budget,
            summarizer=self.summarizer,
            strategy=ContextStrategy.SLIDING_WINDOW,
        )
        assert summary is not None or len(result) < len(msgs)