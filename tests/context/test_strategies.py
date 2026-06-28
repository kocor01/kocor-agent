"""测试上下文策略应用器。"""

from __future__ import annotations
import os

from kocor.config import Config
from kocor.context.budget import TokenBudget
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.types import ContextStrategy
from kocor.llm_provider.llm_manager import LlmManager
from kocor.llm_provider.message import Message


class FakeLLM:
    def __init__(self):
        self.summary_text = "这是历史摘要"

    @property
    def provider(self):
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        return Message(role="assistant", content=self.summary_text)


class TestContextStrategyApplier:
    """测试 ContextStrategyApplier。"""

    def setup_method(self):
        LlmManager._client = FakeLLM()
        self.applier = ContextStrategyApplier()

    def teardown_method(self):
        LlmManager.reset()

    def test_default_no_history(self):
        """DEFAULT 策略下空历史应原样返回。"""
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]
        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.DEFAULT,
        )
        assert summary is None
        assert result == msgs

    def test_default_with_long_history(self):
        """DEFAULT 策略下长历史也不截断。"""
        msgs = [Message(role="user", content=f"msg{i}") for i in range(100)]
        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.DEFAULT,
        )
        assert summary is None
        assert len(result) == 100

    def test_sliding_window_truncates(self):
        """SLIDING_WINDOW 策略应截断超出轮次的消息。"""
        applier = ContextStrategyApplier()
        msgs = []
        for i in range(10):
            msgs.extend([
                Message(role="user", content=f"问题{i}"),
                Message(role="assistant", content=f"回答{i}"),
            ])
        result, summary = applier.apply(
            messages=msgs,
            strategy=ContextStrategy.SLIDING_WINDOW,
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
        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.AGGRESSIVE,
        )
        assert summary is not None
        assert len(result) <= 3

    def test_empty_messages(self):
        """空消息列表应返回空。"""
        result, summary = self.applier.apply(
            messages=[],
            strategy=ContextStrategy.SLIDING_WINDOW,
        )
        assert result == []
        assert summary is None

    def test_tight_budget_triggers_truncation(self):
        """token 预算紧张时 SLIDING_WINDOW 应触发截断。"""
        old = os.environ.get("KOCOR_CONTEXT_MAX_TOKENS")
        os.environ["KOCOR_CONTEXT_MAX_TOKENS"] = "500"
        Config.reset()
        try:
            applier = ContextStrategyApplier()
            msgs = [Message(role="user", content=f"msg{i}") for i in range(30)]
            msgs.extend([Message(role="assistant", content=f"ans{i}") for i in range(30)])
            result, summary = applier.apply(
                messages=msgs,
                strategy=ContextStrategy.SLIDING_WINDOW,
            )
            assert summary is not None or len(result) < len(msgs)
        finally:
            if old is None:
                del os.environ["KOCOR_CONTEXT_MAX_TOKENS"]
            else:
                os.environ["KOCOR_CONTEXT_MAX_TOKENS"] = old
            Config.reset()

    # ── TokenBudget 驱动的策略升级 ──────────────────────

    def test_budget_summarize_upgrades_default(self):
        """should_summarize=True 时 DEFAULT 策略应升级为 SLIDING_WINDOW。"""
        budget = TokenBudget(limit=1000, used_prompt=750)  # ratio=0.75
        msgs = []
        for i in range(10):
            msgs.extend([
                Message(role="user", content=f"问题{i}"),
                Message(role="assistant", content=f"回答{i}"),
            ])

        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.DEFAULT,
            token_budget=budget,
        )
        assert summary is not None
        assert len(result) < len(msgs)

    def test_budget_below_summarize_keeps_default(self):
        """should_summarize=False 时 DEFAULT 策略保持不变。"""
        budget = TokenBudget(limit=1000, used_prompt=500)  # ratio=0.50
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]

        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.DEFAULT,
            token_budget=budget,
        )
        assert summary is None
        assert len(result) == 2

    def test_budget_truncate_overrides_sliding(self):
        """should_truncate=True 时 SLIDING_WINDOW 应降级为 AGGRESSIVE。"""
        budget = TokenBudget(limit=1000, used_prompt=950)  # ratio=0.95
        applier = ContextStrategyApplier()
        msgs = []
        for i in range(8):
            msgs.extend([
                Message(role="user", content=f"问题{i}"),
                Message(role="assistant", content=f"回答{i}"),
            ])

        result, summary = applier.apply(
            messages=msgs,
            strategy=ContextStrategy.SLIDING_WINDOW,
            token_budget=budget,
        )
        assert summary is not None
        # AGGRESSIVE 只保留最后一轮 → 最多 3 条消息（user + assistant + 摘要）
        assert len(result) <= 3

    def test_budget_truncate_overrides_default(self):
        """should_truncate=True 时 DEFAULT 应直接升级为 AGGRESSIVE。"""
        budget = TokenBudget(limit=1000, used_prompt=950)  # ratio=0.95
        msgs = []
        for i in range(8):
            msgs.extend([
                Message(role="user", content=f"问题{i}"),
                Message(role="assistant", content=f"回答{i}"),
            ])

        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.DEFAULT,
            token_budget=budget,
        )
        assert summary is not None
        assert len(result) <= 3

    def test_budget_none_preserves_existing_behavior(self):
        """token_budget=None 时行为不变。"""
        msgs = [Message(role="user", content=f"msg{i}") for i in range(100)]
        result, summary = self.applier.apply(
            messages=msgs,
            strategy=ContextStrategy.DEFAULT,
            token_budget=None,
        )
        assert summary is None
        assert len(result) == 100
