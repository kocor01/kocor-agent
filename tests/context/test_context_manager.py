"""测试 ContextManager 会话管理和上下文压缩。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.llm_provider.message import Message


def _override_config(values: dict) -> dict:
    """覆盖 Config 值，返回原值字典用于恢复。"""
    cfg = Config.load()
    orig = {}
    for key, val in values.items():
        orig[key] = getattr(cfg, key)
        setattr(cfg, key, val)
    return orig


def _restore_config(orig: dict) -> None:
    """恢复 Config 原始值。"""
    cfg = Config.load()
    for key, val in orig.items():
        setattr(cfg, key, val)


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
        _orig = _override_config({
            "context_strategy": "sliding",
            "preserve_last_rounds": 2,
            "preserve_first_rounds": 1,
            "context_max_tokens": 100_000,
            "context_summary_threshold": 0.5,
            "context_truncate_threshold": 0.9,
            "default_system_prompt": "你是一个助手",
        })
        ctx = ContextManager()

        ctx.messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        original_len = len(ctx.messages)
        ctx.compress_if_needed()
        assert len(ctx.messages) == original_len
        _restore_config(_orig)

    def test_compress_if_needed_over_threshold(self):
        """超阈值时压缩应减少消息数量。"""
        with _patch_llm():
            _orig = _override_config({
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            })
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
            _restore_config(_orig)


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
            _orig = _override_config({
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            })
            ctx = ContextManager(tools=FakeToolRegistry())

            # 模拟多轮对话
            for i in range(5):
                ctx.build_initial_context(f"第{i}轮问题")
                ctx.append(Message(role="assistant", content=f"第{i}轮回答"))
                ctx.extract_session_history()

            # extract 不压缩，只过滤
            assert all(m.role != "system" for m in ctx.session_history)
            _restore_config(_orig)

    def test_summary_persists_through_extract(self):
        """通过 build_initial_context(压缩) → extract 后，摘要以 assistant role 存在于 session_history。"""
        with _patch_llm():
            _orig = _override_config({
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            })
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
            _restore_config(_orig)

    def test_compress_if_needed_uses_assistant_role(self):
        """compress_if_needed 压缩后，摘要以 assistant role 存在。"""
        with _patch_llm():
            _orig = _override_config({
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            })
            ctx = ContextManager(tools=FakeToolRegistry())

            messages = [Message(role="system", content="sys")]
            messages.extend(self._make_long_history(10))
            ctx.messages = messages

            ctx.count_message_tokens = MagicMock(return_value=95_000)
            ctx.count_tool_tokens = MagicMock(return_value=50_000)

            ctx.compress_if_needed()

            # 压缩后 messages[0] 应为 system
            assert ctx.messages[0].role == "system"
            _restore_config(_orig)

    def test_compress_if_needed_over_threshold_with_sliding(self):
        """sliding 策略下超阈值压缩应减少消息。"""
        with _patch_llm():
            _orig = _override_config({
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            })
            ctx = ContextManager(tools=FakeToolRegistry())

            messages = [Message(role="system", content="sys")]
            messages.extend(self._make_long_history(10))
            ctx.messages = messages

            ctx.count_message_tokens = MagicMock(return_value=95_000)
            ctx.count_tool_tokens = MagicMock(return_value=50_000)

            ctx.compress_if_needed()

            assert len(ctx.messages) < len(messages)
            _restore_config(_orig)
class TestCompressBudgetConsistency:
    """测试 compress_if_needed 预算口径一致性（P1.3）。

    验证：
    1. 复用 self.token_budget 而非新建 TokenBudget() 实例
    2. 仅用 prompt 侧 token 做压缩判断（不含 completion_tokens）
    3. 有 usage 时优先使用 usage.prompt_tokens，无 usage 时回退到 count_message_tokens()
    """

    def test_compress_if_needed_uses_same_budget(self):
        """应使用 self.token_budget 而非新建实例。"""
        ctx = ContextManager()
        original_budget = ctx.token_budget
        assert original_budget is not None

        ctx.compress_if_needed()

        assert ctx.token_budget is original_budget

    def test_compress_if_needed_default_strategy_skips_noop(self):
        """DEFAULT 策略下即使超阈值也不改变消息（非 SLIDING 策略）。"""
        _orig = _override_config({
            "context_max_tokens": 10000,
            "context_summary_threshold": 0.5,
        })
        ctx = ContextManager()
        ctx.messages = [Message(role="user", content="hi")]

        ctx.usage = MagicMock()
        ctx.usage.prompt_tokens = 100000      # 远超阈值
        ctx.usage.completion_tokens = 0

        ctx.compress_if_needed()

        # DEFAULT 策略不截断，消息数不变
        assert len(ctx.messages) == 1
        _restore_config(_orig)

    def test_compress_if_needed_uses_api_usage_when_available(self):
        """有 usage 时使用 usage.prompt_tokens + usage.completion_tokens，不调用本地估算。"""
        with _patch_llm():
            _orig = _override_config({
                "context_strategy": "sliding",
                "context_max_tokens": 1000,
                "context_summary_threshold": 0.3,
            })
            ctx = ContextManager()
            ctx.messages = [Message(role="system", content="sys")]
            for i in range(10):
                ctx.messages.append(Message(role="user", content=f"q{i}"))
                ctx.messages.append(Message(role="assistant", content=f"a{i}"))

            ctx.usage = MagicMock()
            ctx.usage.prompt_tokens = 200       # 20%
            ctx.usage.completion_tokens = 200    # 20% → 合计 40% > 30% → 触发压缩
            ctx.count_message_tokens = MagicMock(return_value=50)  # 5% — 不应被用到

            ctx.compress_if_needed()

            # 有 usage 时 total=400, ratio=0.4>0.3 → 触发压缩
            assert len(ctx.messages) < 21
            _restore_config(_orig)

    def test_compress_if_needed_without_usage(self):
        """无 usage 时回退到 count_message_tokens() + count_tool_tokens()。"""
        _orig = _override_config({
            "context_max_tokens": 1000,
            "context_summary_threshold": 0.3,
        })
        ctx = ContextManager()
        ctx.messages = [Message(role="user", content="hi")]
        ctx.usage = None  # 无 API 返回的 usage

        called = [False]

        def _track_call(*args, **kwargs):
            called[0] = True
            return 50  # 极低 token，不会触发压缩

        original = ctx.count_message_tokens
        ctx.count_message_tokens = _track_call

        try:
            ctx.compress_if_needed()
        finally:
            ctx.count_message_tokens = original

        assert called[0], "count_message_tokens should be called when usage is None"
        _restore_config(_orig)

    def test_compress_uses_api_usage_sum(self):
        """有 usage 时 total = prompt_tokens + completion_tokens 做压缩判断。"""
        with _patch_llm():
            _orig = _override_config({
                "context_strategy": "sliding",
                "preserve_last_rounds": 2,
                "preserve_first_rounds": 1,
                "context_max_tokens": 100_000,
                "context_summary_threshold": 0.5,
                "context_truncate_threshold": 0.9,
                "default_system_prompt": "你是一个助手",
            })
            ctx = ContextManager()

            ctx.messages = [Message(role="system", content="sys")]
            for i in range(10):
                ctx.messages.append(Message(role="user", content=f"q{i}"))
                ctx.messages.append(Message(role="assistant", content=f"a{i}"))

            # 场景 A：usage 的 prompt+completion 低 → 不压缩
            ctx.usage = MagicMock()
            ctx.usage.prompt_tokens = 1000       # 1%
            ctx.usage.completion_tokens = 1000   # 1% → total 2% < 50%

            ctx.count_message_tokens = MagicMock(return_value=95_000)  # 不应被用到
            ctx.count_tool_tokens = MagicMock(return_value=0)

            ctx.compress_if_needed()
            assert len(ctx.messages) == 21, "usage 合计低时不应压缩"

            # 场景 B：usage 的 prompt+completion 高 → 压缩
            ctx.usage = MagicMock()
            ctx.usage.prompt_tokens = 50_000     # 50%
            ctx.usage.completion_tokens = 50_000  # 50% → total 100% > 50%

            ctx.compress_if_needed()
            assert len(ctx.messages) < 21, "usage 合计超阈值时应压缩"

            _restore_config(_orig)

