"""测试 ContextManager 会话管理和上下文压缩。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kocor.context.budget import TokenBudget
from kocor.context.context_manager import ContextManager
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.types import ContextStrategy
from kocor.llm_provider.message import Message


class FakeToolRegistry:
    def get_definitions(self):
        return []


class TestContextManager:
    """测试 ContextManager 基本功能。"""

    def test_default_initial_state(self):
        """空构造应提供合理的默认值。"""
        ctx = ContextManager()
        assert ctx.messages == []
        assert ctx.session_history == []
        assert ctx.iteration == 0
        assert ctx.token_budget is not None
        assert ctx.system_content == ""

    def test_reset_clears_iteration_state(self):
        """reset() 应清空本轮状态但保留 session_history。"""
        ctx = ContextManager()
        ctx.messages = [Message(role="user", content="hi")]
        ctx.iteration = 5
        ctx.session_history = [Message(role="assistant", content="prev")]

        ctx.reset()

        assert ctx.messages == []
        assert ctx.iteration == 0
        # session_history 应保留
        assert len(ctx.session_history) == 1

    def test_reset_conversation_clears_all(self):
        """reset_conversation() 应清空所有状态包括 session_history。"""
        ctx = ContextManager()
        ctx.session_history = [Message(role="assistant", content="prev")]
        ctx.iteration = 3

        ctx.reset_conversation()

        assert ctx.session_history == []
        assert ctx.iteration == 0

    def test_append_adds_message(self):
        """append() 应追加消息到 messages。"""
        ctx = ContextManager()
        ctx.append(Message(role="user", content="hi"))
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "hi"

    def test_extract_session_history(self):
        """extract_session_history() 应提取非 system 消息。"""
        ctx = ContextManager()
        ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="tool", content="result", tool_call_id="c1"),
        ]
        ctx.extract_session_history()
        assert len(ctx.session_history) == 3
        assert all(m.role != "system" for m in ctx.session_history)

    def test_advance_iteration(self):
        """advance_iteration() 应递增计数。"""
        ctx = ContextManager()
        ctx.advance_iteration()
        assert ctx.iteration == 1
        ctx.advance_iteration()
        assert ctx.iteration == 2


class TestContextManagerCompression:
    """测试上下文压缩功能。"""

    def test_compress_if_needed_default_skips(self):
        """DEFAULT 策略不压缩。"""
        with patch("kocor.context.context_manager.Config") as mock_config:
            mock_config.get.return_value = "default"
            ctx = ContextManager()
        ctx.messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="x" * 100_000),
            Message(role="assistant", content="y" * 100_000),
        ]
        original_len = len(ctx.messages)
        ctx.compress_if_needed()
        assert len(ctx.messages) == original_len

    def test_compress_if_needed_within_budget(self):
        """未超阈值时不压缩。"""
        with patch("kocor.context.context_manager.Config") as mock_config:
            mock_config.get.return_value = "sliding"
            ctx = ContextManager()
        ctx.messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        original_len = len(ctx.messages)
        ctx.compress_if_needed()
        assert len(ctx.messages) == original_len

    def test_compress_if_needed_over_threshold(self):
        """超阈值时压缩应减少消息数量。"""
        with patch("kocor.context.context_manager.Config") as mock_config:
            mock_config.get.return_value = "sliding"
            ctx = ContextManager()

            # mock token 计数，让 compress_if_needed 认为超阈值
            ctx.count_message_tokens = MagicMock(return_value=95_000)
            ctx.count_tool_tokens = MagicMock(return_value=50_000)

            messages = [Message(role="system", content="sys")]
            for i in range(10):
                messages.append(Message(role="user", content=f"q{i}"))
                messages.append(Message(role="assistant", content=f"a{i}"))
            ctx.messages = messages

            ctx.compress_if_needed()

            assert len(ctx.messages) < len(messages)

    def test_build_initial_context_uses_session_history(self):
        """build_initial_context 应使用 session_history 构建消息列表。"""
        with patch("kocor.context.context_manager.Config") as mock_config:
            mock_config.get.return_value = "default"
            ctx = ContextManager(
                identity_prompt="test",
                tools=FakeToolRegistry(),
            )
        ctx.session_history = [Message(role="user", content="prev_q"),
                                Message(role="assistant", content="prev_a")]

        ctx.build_initial_context("new_q")

        assert len(ctx.messages) == 4  # system + 2 history + current input
        assert ctx.messages[1].content == "prev_q"
        assert ctx.messages[3].content == "new_q"
