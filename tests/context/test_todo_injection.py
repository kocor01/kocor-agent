"""测试 ContextManager 在上下文压缩时注入 active todos 快照。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.llm_provider.message import Message
from kocor.tools.toolset.todo_tool import TodoStore


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


_CONFIG = {
    "context_strategy": "sliding",
    "preserve_last_rounds": 2,
    "preserve_first_rounds": 1,
    "context_max_tokens": 100_000,
    "context_summary_threshold": 0.5,
    "context_truncate_threshold": 0.9,
    "default_system_prompt": "你是一个助手",
}


def _make_long_history(n_rounds: int = 8) -> list[Message]:
    msgs = []
    for i in range(n_rounds):
        msgs.append(Message(role="user", content=f"q{i}"))
        msgs.append(Message(role="assistant", content=f"a{i}"))
    return msgs


def _patch_llm():
    """返回一个上下文管理器，将 LlmFactory.create 替换为 FakeLLMForSummary。"""
    return patch(
        "kocor.llm_provider.llm_factory.LlmFactory.create",
        return_value=FakeLLMForSummary(),
    )


class TestTodoSnapshotInjection:
    """测试上下文压缩时注入 active todos 快照。"""

    def test_build_initial_context_injects_on_compression(self):
        """压缩发生（summary_node 非空）时，应在 user_input 前注入 active todos。"""
        with _patch_llm():
            _orig = _override_config(_CONFIG)
            store = TodoStore()
            store.write([{"id": "1", "content": "active task", "status": "in_progress"}])
            ctx = ContextManager(tools=FakeToolRegistry(), todo_store=store)

            ctx.session_history = _make_long_history(8)
            ctx.build_initial_context("最新问题")

            injected = [m for m in ctx.messages if m.role == "user" and "active task" in m.content]
            assert len(injected) == 1
            # 注入在最后一条 user_input 之前
            assert ctx.messages[-1].content == "最新问题"
            _restore_config(_orig)

    def test_build_initial_context_no_injection_without_compression(self):
        """未压缩（summary_node 为 None）时不注入。"""
        _orig = _override_config({
            "context_strategy": "default",
            "default_system_prompt": "你是一个助手",
        })
        store = TodoStore()
        store.write([{"id": "1", "content": "active", "status": "pending"}])
        ctx = ContextManager(tools=FakeToolRegistry(), todo_store=store)

        ctx.session_history = [Message(role="user", content="prev")]
        ctx.build_initial_context("new")
        injected = [m for m in ctx.messages if "active" in m.content and m.role == "user"]
        assert injected == []
        _restore_config(_orig)

    def test_build_initial_context_no_injection_when_no_active(self):
        """无 active 项时不注入（即使压缩发生）。"""
        with _patch_llm():
            _orig = _override_config(_CONFIG)
            store = TodoStore()
            store.write([{"id": "1", "content": "done", "status": "completed"}])
            ctx = ContextManager(tools=FakeToolRegistry(), todo_store=store)

            ctx.session_history = _make_long_history(8)
            ctx.build_initial_context("最新问题")
            injected = [m for m in ctx.messages if "preserved across context compression" in m.content]
            assert injected == []
            _restore_config(_orig)

    def test_no_todo_store_never_injects(self):
        """未提供 todo_store 时即使压缩也不注入。"""
        with _patch_llm():
            _orig = _override_config(_CONFIG)
            ctx = ContextManager(tools=FakeToolRegistry())  # 无 todo_store

            ctx.session_history = _make_long_history(8)
            ctx.build_initial_context("最新问题")
            injected = [m for m in ctx.messages if "preserved across context compression" in m.content]
            assert injected == []
            _restore_config(_orig)

    def test_compress_if_needed_injects_snapshot(self):
        """compress_if_needed 压缩后应在末尾追加快照。"""
        with _patch_llm():
            _orig = _override_config(_CONFIG)
            store = TodoStore()
            store.write([{"id": "1", "content": "active task", "status": "in_progress"}])
            ctx = ContextManager(tools=FakeToolRegistry(), todo_store=store)

            messages = [Message(role="system", content="sys")]
            messages.extend(_make_long_history(10))
            ctx.messages = messages
            ctx.count_message_tokens = MagicMock(return_value=95_000)
            ctx.count_tool_tokens = MagicMock(return_value=50_000)

            ctx.compress_if_needed()

            injected = [m for m in ctx.messages if "preserved across context compression" in m.content]
            assert len(injected) == 1
            assert "active task" in injected[0].content
            _restore_config(_orig)