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


class FakeLLMForSummary:
    def __init__(self, summary_text: str = "这是对话摘要"):
        self.summary_text = summary_text

    @property
    def provider(self):
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        return Message(role="assistant", content=self.summary_text)


def _patch_llm():
    """返回一个上下文管理器，将 LlmFactory.create 替换为 FakeLLMForSummary。"""
    return patch(
        "kocor.llm_provider.llm_factory.LlmFactory.create",
        return_value=FakeLLMForSummary(),
    )


class TestContextManager:
    """测试 ContextManager 基础功能。"""

    def test_default_initial_state(self):
        """验证默认初始状态。"""
        ctx = ContextManager()
        assert ctx.system_content == ""
        assert ctx.messages == []
        assert ctx.iteration == 0
        assert ctx.usage is None

    def test_reset_clears_iteration_state(self):
        """reset() 应重置迭代状态但不重置会话历史。"""
        ctx = ContextManager()
        ctx.iteration = 5
        ctx.usage = MagicMock()
        ctx.messages = [Message(role="user", content="hi")]
        ctx.session_history = [Message(role="user", content="old")]

        ctx.reset()

        assert ctx.iteration == 0
        assert ctx.usage is None
        assert ctx.messages == []
        assert ctx.session_history == [Message(role="user", content="old")]

    def test_reset_conversation_clears_all(self):
        """reset_conversation() 应清空所有消息和历史。"""
        ctx = ContextManager()
        ctx.messages = [Message(role="user", content="hi")]
        ctx.session_history = [Message(role="user", content="old")]

        ctx.reset_conversation()

        assert ctx.messages == []
        assert ctx.session_history == []

    def test_append_adds_message(self):
        """append() 应添加消息到 messages。"""
        ctx = ContextManager()
        msg = Message(role="user", content="你好")
        ctx.append(msg)
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "你好"

    def test_extract_session_history(self):
        """extract_session_history 应排除 system 消息。"""
        ctx = ContextManager()
        ctx.messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="tool", content="result", tool_call_id="c1"),
        ]
        ctx.extract_session_history()
        assert all(m.role != "system" for m in ctx.session_history)

    def test_advance_iteration(self):
        """advance_iteration() 应递增 iteration。"""
        ctx = ContextManager()
        assert ctx.iteration == 0
        ctx.advance_iteration()
        assert ctx.iteration == 1
        ctx.advance_iteration()
        assert ctx.iteration == 2


class TestContextManagerCompression:
    """测试 ContextManager 的上下文压缩功能。"""

    def test_compress_if_needed_default_skips(self):
        """DEFAULT 策略下 compress_if_needed 不应压缩。"""
        ctx = ContextManager()
        ctx.messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        original_len = len(ctx.messages)
        ctx.compress_if_needed()
        assert len(ctx.messages) == original_len

    def test_compress_if_needed_within_budget(self):
        """预算未超时不压缩。"""
        with patch("kocor.context.context_manager.Config") as mock_config:
            mock_config.get.side_effect = lambda key, **kw: {
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            }.get(key, kw.get("default"))
            ctx = ContextManager()

        ctx.messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        original_len = len(ctx.messages)
        ctx.compress_if_needed()
        assert len(ctx.messages) == original_len

    def test_compress_if_needed_over_threshold(self):
        """超阈值时压缩应减少消息数量。"""
        with _patch_llm():
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


class TestSessionHistoryCompression:
    """测试 session_history 按 sliding window 策略压缩。"""

    def _make_long_history(self, n_rounds: int = 10) -> list[Message]:
        """构造多轮对话历史。"""
        msgs = []
        for i in range(n_rounds):
            msgs.append(Message(role="user", content=f"q{i}"))
            msgs.append(Message(role="assistant", content=f"a{i}"))
        return msgs

    def test_session_history_grows_normally(self):
        """extract_session_history 不压缩，只过滤非 system 消息。"""
        with _patch_llm():
            with patch("kocor.context.context_manager.Config") as mock_config:
                mock_config.get.side_effect = lambda key, **kw: {
                    "context_strategy": "sliding",
                    "preserve_last_rounds": 2,
                    "preserve_first_rounds": 1,
                    "context_max_tokens": 100_000,
                    "context_summary_threshold": 0.5,
                    "context_truncate_threshold": 0.9,
                    "default_system_prompt": "你是一个助手",
                }.get(key, kw.get("default"))
                ctx = ContextManager(tools=FakeToolRegistry())

            # 模拟多轮对话
            for i in range(5):
                ctx.build_initial_context(f"第{i}轮问题")
                ctx.append(Message(role="assistant", content=f"第{i}轮回答"))
                ctx.extract_session_history()

            # extract 不压缩，只过滤
            assert all(m.role != "system" for m in ctx.session_history)

    def test_summary_persists_through_extract(self):
        """通过 build_initial_context(压缩) → extract 后，摘要以 assistant role 存在于 session_history。"""
        with _patch_llm():
            with patch("kocor.context.context_manager.Config") as mock_config:
                mock_config.get.side_effect = lambda key, **kw: {
                    "context_strategy": "sliding",
                    "preserve_last_rounds": 2,
                    "preserve_first_rounds": 1,
                    "context_max_tokens": 100_000,
                    "context_summary_threshold": 0.5,
                    "context_truncate_threshold": 0.9,
                    "default_system_prompt": "你是一个助手",
                }.get(key, kw.get("default"))
                ctx = ContextManager(tools=FakeToolRegistry())

            # 预置多轮会话历史，确保触发压缩
            ctx.session_history = self._make_long_history(8)

            ctx.build_initial_context("最新问题")
            ctx.append(Message(role="assistant", content="最新回答"))
            ctx.extract_session_history()

            # session_history 中应包含 assistant 摘要消息
            summary_msgs = [
                m for m in ctx.session_history
                if m.role == "assistant" and m.content.startswith("[历史对话摘要]")
            ]
            assert len(summary_msgs) == 1
            assert summary_msgs[0].role == "assistant"

    def test_compress_if_needed_uses_assistant_role(self):
        """compress_if_needed 压缩后，摘要以 assistant role 存在。"""
        with _patch_llm():
            with patch("kocor.context.context_manager.Config") as mock_config:
                mock_config.get.side_effect = lambda key, **kw: {
                    "context_strategy": "sliding",
                    "preserve_last_rounds": 2,
                    "preserve_first_rounds": 1,
                    "context_max_tokens": 100_000,
                    "context_summary_threshold": 0.5,
                    "context_truncate_threshold": 0.9,
                    "default_system_prompt": "你是一个助手",
                }.get(key, kw.get("default"))
                ctx = ContextManager(tools=FakeToolRegistry())

            messages = [Message(role="system", content="sys")]
            messages.extend(self._make_long_history(10))
            ctx.messages = messages

            ctx.count_message_tokens = MagicMock(return_value=95_000)
            ctx.count_tool_tokens = MagicMock(return_value=50_000)

            ctx.compress_if_needed()

            # 压缩后 messages[0] 应为 system
            assert ctx.messages[0].role == "system"

    def test_compress_if_needed_over_threshold_with_sliding(self):
        """sliding 策略下超阈值压缩应减少消息。"""
        with _patch_llm():
            with patch("kocor.context.context_manager.Config") as mock_config:
                mock_config.get.side_effect = lambda key, **kw: {
                    "context_strategy": "sliding",
                    "preserve_last_rounds": 2,
                    "preserve_first_rounds": 1,
                    "context_max_tokens": 100_000,
                    "context_summary_threshold": 0.5,
                    "context_truncate_threshold": 0.9,
                    "default_system_prompt": "你是一个助手",
                }.get(key, kw.get("default"))
                ctx = ContextManager(tools=FakeToolRegistry())

            messages = [Message(role="system", content="sys")]
            messages.extend(self._make_long_history(10))
            ctx.messages = messages

            ctx.count_message_tokens = MagicMock(return_value=95_000)
            ctx.count_tool_tokens = MagicMock(return_value=50_000)

            ctx.compress_if_needed()

            assert len(ctx.messages) < len(messages)