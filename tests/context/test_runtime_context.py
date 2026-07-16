"""RuntimeContext 纯数据类 + ContextCompressor 测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from kocor.context.compressor import ContextCompressor
from kocor.context.runtime_context import RuntimeContext
from kocor.context.budget import TokenBudget
from kocor.llm_provider.message import Message


class TestRuntimeContext:
    def test_runtime_context_is_pure_dataclass(self):
        """RuntimeContext 是纯数据类，可直接构造和操作。"""
        ctx = RuntimeContext()
        ctx.messages.append(Message(role="user", content="hi"))
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "hi"
        assert ctx.iteration == 0

    def test_runtime_context_reset(self):
        """reset 应清除 messages 和 iteration，保留 session_history。"""
        ctx = RuntimeContext()
        ctx.messages.append(Message(role="user", content="hi"))
        ctx.session_history.append(Message(role="user", content="old"))
        ctx.iteration = 5
        ctx.reset()
        assert ctx.iteration == 0
        assert len(ctx.messages) == 0
        assert len(ctx.session_history) == 1  # reset 不清除 session_history

    def test_runtime_context_reset_conversation(self):
        """reset_conversation 应清除所有数据。"""
        ctx = RuntimeContext()
        ctx.messages.append(Message(role="user", content="hi"))
        ctx.session_history.append(Message(role="user", content="old"))
        ctx.iteration = 5
        ctx.reset_conversation()
        assert ctx.iteration == 0
        assert len(ctx.messages) == 0
        assert len(ctx.session_history) == 0


class TestContextCompressor:
    def test_compressor_noop_when_below_threshold(self):
        """未达阈值时压缩器不应修改消息。"""
        compressor = ContextCompressor()
        ctx = RuntimeContext()
        ctx.messages = [Message(role="user", content="short")]
        ctx.token_budget = TokenBudget(
            limit=100_000, threshold_summary=0.9, threshold_truncate=1.0,
        )
        compressor.compress_if_needed(ctx, todo_store=None)
        assert len(ctx.messages) == 1

    def test_compressor_with_external_total_token(self):
        """传入 total_token 时 compressor 应据此判断。"""
        compressor = ContextCompressor()
        ctx = RuntimeContext()
        ctx.messages = [Message(role="user", content="short")]
        ctx.token_budget = TokenBudget(
            limit=100, threshold_summary=0.5, threshold_truncate=0.9,
        )
        # total_token=90 > 阈值 50，但消息太少无法压缩，不应崩溃
        compressor.compress_if_needed(ctx, todo_store=None, total_token=90)
        # 不应抛出异常
        assert len(ctx.messages) >= 1