"""Loop 引擎直接单元测试 — 不通过 Agent 集成，直接测试 Loop 的边界情况和辅助方法。

覆盖代码审查报告指出的「重复检测逻辑、stop() 行为、预算耗尽处理缺少隔离测试」缺口。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from kocor.event.event_manager import EventEmitter, EventType
from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.message import FunctionCall, Message, ToolCall, StreamChunk


# ── Helper: 构建 Loop 所需的最小 mock ──


class MockContext:
    """模拟 ContextManager，暴露 Loop 访问的属性。"""

    def __init__(self):
        self.messages: list[Message] = []
        self.iteration = 0
        self.usage = None
        self.session_history: list[Message] = []

    def reset(self):
        self.messages = []
        self.iteration = 0
        self.usage = None

    def advance_iteration(self):
        self.iteration += 1

    def append(self, msg: Message):
        self.messages.append(msg)

    def compress_if_needed(self):
        pass

    def extract_session_history(self):
        self.session_history = list(self.messages)

    def build_initial_context(self, user_input: str):
        self.messages = [
            Message(role="system", content="system"),
            Message(role="user", content=user_input),
        ]


class MockLLM:
    """模拟 LLM，非 Agent 集成版本。"""

    def __init__(self, responses: list[Message] | None = None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, messages, tools=None):
        resp = self.responses[self.call_count] if self.call_count < len(self.responses) else Message(role="assistant", content="done")
        self.call_count += 1
        return resp

    def stream(self, messages, tools=None):
        resp = self.generate(messages, tools)
        if resp.content:
            yield StreamChunk(content=resp.content, is_final=False)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                yield StreamChunk(tool_calls=[tc], is_final=False)
        yield StreamChunk(is_final=True)

    @property
    def provider(self):
        return "fake"


class MockToolRegistry:
    skill_manager = None

    def __init__(self):
        self.executed = []

    def get_definitions(self, filter_category=None):
        return []

    def execute(self, tool_call):
        self.executed.append(tool_call)
        from kocor.llm_provider.message import ToolResult
        return ToolResult(
            tool_call_id=tool_call.id,
            content=f"Result of {tool_call.function.name}",
        )

    def start_cron_scheduler(self):
        pass

    def stop_cron_scheduler(self):
        pass


def _make_loop(llm_responses=None, max_iterations=10):
    """创建最小 Loop 实例。"""
    from kocor.loop import Loop

    llm = MockLLM(responses=llm_responses or [Message(role="assistant", content="ok")])
    ctx = MockContext()
    tool_mgr = MockToolRegistry()
    pm = MagicMock()
    pm.check.return_value = True
    hm = HookManager()
    ee = EventEmitter()
    return Loop(
        llm=llm,
        ctx=ctx,
        tool_manager=tool_mgr,
        permission_mgr=pm,
        hook_manager=hm,
        event_emitter=ee,
        max_iterations=max_iterations,
    )


# ═══════════════════════════════════════════════
# 重复工具调用检测 — _check_repetition 直接测试
# ═══════════════════════════════════════════════


class TestCheckRepetitionDirect:
    """直接测试 _check_repetition 方法，不经过完整循环。"""

    def test_no_tool_calls_resets_state(self):
        """没有工具调用时重置计数器并返回 False。"""
        loop = _make_loop()
        loop._consecutive_duplicate_count = 3
        loop._last_tool_call_signature = "something"

        response = Message(role="assistant", content="text only")
        result = loop._check_repetition(response)

        assert result is False
        assert loop._consecutive_duplicate_count == 0
        assert loop._last_tool_call_signature is None

    def test_first_call_no_duplicate(self):
        """首次工具调用设置计数为 1，不触发。"""
        loop = _make_loop()
        response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )
        result = loop._check_repetition(response)

        assert result is False
        assert loop._consecutive_duplicate_count == 1

    def test_second_duplicate_injects_warning(self):
        """第 2 次重复注入显式警告消息到上下文，不返回 True。"""
        loop = _make_loop()
        loop._consecutive_duplicate_count = 1
        loop._last_tool_call_signature = 'search({"q": "test"})'

        response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )
        result = loop._check_repetition(response)

        assert result is False
        assert loop._consecutive_duplicate_count == 2

        # 验证警告消息被注入
        warning_msgs = [m for m in loop.ctx.messages if "重复的工具调用" in m.content]
        assert len(warning_msgs) == 1

    def test_third_duplicate_returns_true_and_emits_event(self):
        """第 3 次重复返回 True 并触发 ON_BUDGET_EXHAUSTED 事件和钩子。"""
        loop = _make_loop()
        loop._consecutive_duplicate_count = 2
        loop._last_tool_call_signature = 'search({"q": "test"})'

        # 注册钩子跟踪
        hook_calls = []
        hook_manager = loop.hook_manager

        class TrackHook:
            hook_point = HookPoint.ON_BUDGET_EXHAUSTED
            def run(self, ctx):
                hook_calls.append(ctx.iteration)
                return HookResult(action=HookAction.CONTINUE)

        hook_manager.register(TrackHook())

        # 注册事件跟踪
        events = []
        loop.event_emitter.subscribe(EventType.ON_BUDGET_EXHAUSTED, lambda e: events.append(e))

        response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )
        result = loop._check_repetition(response)

        assert result is True
        assert loop._consecutive_duplicate_count == 3

        # 验证事件触发
        assert len(events) >= 1
        assert events[0].type == EventType.ON_BUDGET_EXHAUSTED
        assert events[0].data.get("reason") == "duplicate_tool_calls"

        # 验证钩子触发
        assert len(hook_calls) >= 1

    def test_count_increments_from_zero_across_calls(self):
        """从 0→1→2→3 跟踪完整递增。"""
        loop = _make_loop()
        response = Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="search", arguments='{"q": "t"}'))],
        )

        # 第 1 次
        r1 = loop._check_repetition(response)
        assert r1 is False and loop._consecutive_duplicate_count == 1

        # 第 2 次（注入警告）
        r2 = loop._check_repetition(response)
        assert r2 is False and loop._consecutive_duplicate_count == 2

        # 第 3 次（触发检测）
        r3 = loop._check_repetition(response)
        assert r3 is True and loop._consecutive_duplicate_count == 3

    def test_different_signature_resets_count(self):
        """不同签名的工具调用重置重复计数。"""
        loop = _make_loop()
        loop._consecutive_duplicate_count = 2
        loop._last_tool_call_signature = 'search({"q": "old"})'

        response = Message(
            role="assistant",
            tool_calls=[ToolCall(id="c2", function=FunctionCall(name="search", arguments='{"q": "new"}'))],
        )
        result = loop._check_repetition(response)

        assert result is False
        # 不同的参数 → 重置到 1
        assert loop._consecutive_duplicate_count == 1
        assert loop._last_tool_call_signature == 'search({"q": "new"})'

    def test_multiple_tool_calls_combined_signature(self):
        """多个工具调用的签名是全部合并后的字符串。"""
        loop = _make_loop()
        response = Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="c1", function=FunctionCall(name="read", arguments='{"path": "a.txt"}')),
                ToolCall(id="c2", function=FunctionCall(name="read", arguments='{"path": "b.txt"}')),
            ],
        )
        result = loop._check_repetition(response)

        assert result is False
        assert loop._consecutive_duplicate_count == 1
        # 两个工具调用用 | 连接
        assert 'read({' in loop._last_tool_call_signature
        assert "|" in loop._last_tool_call_signature


# ═══════════════════════════════════════════════
# _get_tool_call_signature 边界情况
# ═══════════════════════════════════════════════


class TestGetToolCallSignatureEdgeCases:
    """_get_tool_call_signature 的边界情况。"""

    def test_invalid_json_arguments(self):
        """arguments 为非法 JSON 时保持原样。"""
        from kocor.loop import Loop

        tc = ToolCall(id="1", function=FunctionCall(name="test", arguments="not valid json"))
        sig = Loop._get_tool_call_signature(tc)
        assert "test(" in sig
        assert "not valid json" in sig

    def test_empty_arguments(self):
        """空 arguments 字符串被正确处理。"""
        from kocor.loop import Loop

        tc = ToolCall(id="1", function=FunctionCall(name="test", arguments=""))
        sig = Loop._get_tool_call_signature(tc)
        assert sig == 'test("")'

    def test_none_arguments(self):
        """arguments 为空字典时正常。"""
        from kocor.loop import Loop

        tc = ToolCall(id="1", function=FunctionCall(name="test", arguments="{}"))
        sig = Loop._get_tool_call_signature(tc)
        assert sig == 'test({})'

    def test_different_arg_orders_same_signature(self):
        """参数顺序不同应生成相同签名（通过 sort_keys）。"""
        from kocor.loop import Loop

        tc1 = ToolCall(id="1", function=FunctionCall(name="search", arguments='{"a": 1, "b": 2}'))
        tc2 = ToolCall(id="2", function=FunctionCall(name="search", arguments='{"b": 2, "a": 1}'))
        assert Loop._get_tool_call_signature(tc1) == Loop._get_tool_call_signature(tc2)

    def test_unicode_arguments(self):
        """Unicode 参数正确处理。"""
        from kocor.loop import Loop

        tc = ToolCall(id="1", function=FunctionCall(name="search", arguments='{"q": "中文测试"}'))
        sig = Loop._get_tool_call_signature(tc)
        assert "中文测试" in sig

    def test_nested_dict_arguments(self):
        """嵌套字典参数正常。"""
        from kocor.loop import Loop

        tc = ToolCall(id="1", function=FunctionCall(name="complex", arguments='{"filter": {"x": 1, "y": [2, 3]}}'))
        sig = Loop._get_tool_call_signature(tc)
        assert "complex({" in sig
        assert "filter" in sig


# ═══════════════════════════════════════════════
# 辅助消息格式化
# ═══════════════════════════════════════════════


class TestLoopMessageFormatting:
    """_budget_exhausted_message、_stuck_in_loop_message、_stopped_message。"""

    def test_budget_exhausted_message_format(self):
        loop = _make_loop()
        loop.ctx.iteration = 5
        msg = loop._budget_exhausted_message()
        assert "5" in msg
        assert "迭代" in msg

    def test_stuck_in_loop_message_format(self):
        loop = _make_loop()
        loop.ctx.iteration = 3
        loop._consecutive_duplicate_count = 3
        msg = loop._stuck_in_loop_message()
        assert "3" in msg
        assert "重复" in msg

    def test_stopped_message_format(self):
        loop = _make_loop()
        loop.ctx.iteration = 2
        msg = loop._stopped_message()
        assert "2" in msg
        assert "终止" in msg

    def test_stopped_message_resets_stop_flag(self):
        """_stopped_message 应将 _stop_requested 重置为 False。"""
        loop = _make_loop()
        loop._stop_requested = True
        loop._stopped_message()
        assert loop._stop_requested is False


# ═══════════════════════════════════════════════
# _execute_one_tool 边界情况
# ═══════════════════════════════════════════════


class TestExecuteOneToolEdgeCases:

    def test_permission_denied_returns_error_message(self):
        """权限拒绝时返回 Permission Denied 消息给 LLM。"""
        loop = _make_loop()
        loop.permission_mgr.check.return_value = False

        tc = ToolCall(id="call_1", function=FunctionCall(name="write_file", arguments='{"path": "/etc/passwd"}'))
        result = loop._execute_one_tool(tc)

        assert result is not None
        assert result.role == "tool"
        assert "Permission Denied" in result.content

    def test_skip_by_hook_returns_skip_message(self):
        """钩子返回 ABORT 时跳过工具执行。"""
        loop = _make_loop()
        hook_manager = loop.hook_manager

        class SkipHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.ABORT, message="skip this tool")

        hook_manager.register(SkipHook())

        tc = ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'))
        result = loop._execute_one_tool(tc)

        assert result is not None
        assert result.role == "tool"
        assert "skip this tool" in result.content

    def test_tool_execution_error_returns_error_message(self):
        """工具执行异常时返回错误消息给 LLM。"""

        class ErrorToolRegistry:
            skill_manager = None
            def get_definitions(self, filter_category=None):
                return []
            def execute(self, tool_call):
                raise RuntimeError("connection timeout")
            def start_cron_scheduler(self):
                pass
            def stop_cron_scheduler(self):
                pass

        loop = _make_loop()
        loop.tool_manager = ErrorToolRegistry()

        tc = ToolCall(id="call_e1", function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'))
        result = loop._execute_one_tool(tc)

        assert result is not None
        assert result.role == "tool"
        assert "RuntimeError" in result.content
        assert "connection timeout" in result.content

    def test_successful_tool_execution(self):
        """工具执行成功返回 result 消息。"""
        loop = _make_loop()
        tc = ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'))
        result = loop._execute_one_tool(tc)

        assert result is not None
        assert result.role == "tool"
        assert "Result of read_file" in result.content


# ═══════════════════════════════════════════════
# _reset_state 行为
# ═══════════════════════════════════════════════


class TestResetState:

    def test_reset_state_clears_all(self):
        """_reset_state 重置所有运行状态。"""
        loop = _make_loop()
        loop.ctx.iteration = 5
        loop._consecutive_duplicate_count = 3
        loop._last_tool_call_signature = "something"
        loop._stop_requested = True
        loop.ctx.messages = [Message(role="user", content="existing")]

        loop._reset_state()

        assert loop.ctx.iteration == 0
        assert loop._consecutive_duplicate_count == 0
        assert loop._last_tool_call_signature is None
        assert loop._stop_requested is False

    def test_reset_state_does_not_clear_session_history(self):
        """_reset_state 不清除 session_history（仅运行时状态）。"""
        loop = _make_loop()
        loop.ctx.session_history = [Message(role="user", content="history")]

        loop._reset_state()

        # session_history 保留
        assert len(loop.ctx.session_history) == 1


# ═══════════════════════════════════════════════
# run_messages / stream_messages 边界
# ═══════════════════════════════════════════════


class TestRunMessagesEdgeCases:

    def test_run_messages_with_empty_messages(self):
        """messages 为空时循环应仍运行（LLM 会收到空列表）。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="response")])
        loop.ctx.messages = []

        result = loop.run_messages()

        assert result == "response"

    def test_run_messages_extracts_session_history(self):
        """run_messages 结束后 session_history 自动填充。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="ok")])
        loop.ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ]

        loop.run_messages()

        assert len(loop.ctx.session_history) >= 1
        roles = [m.role for m in loop.ctx.session_history]
        assert "assistant" in roles

    def test_stream_messages_forwarded(self):
        """stream_messages 逐块产出内容。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="hello world")])
        loop.ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hi"),
        ]

        chunks = list(loop.stream_messages())

        assert len(chunks) > 0
        contents = [c.content for c in chunks if c.content]
        assert "hello world" in "".join(contents)

    def test_stream_messages_extracts_session_history(self):
        """stream_messages 消费完毕后 session_history 自动填充。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="streaming")])
        loop.ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hi"),
        ]

        list(loop.stream_messages())

        assert len(loop.ctx.session_history) >= 1
        assert "streaming" in loop.ctx.session_history[-1].content


# ═══════════════════════════════════════════════
# stop() / budget 耗尽
# ═══════════════════════════════════════════════


class TestStopAndBudgetExhaustion:

    def test_stop_immediately_in_run(self):
        """stop() 后 run_messages 立即返回停止消息。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="should not see")])
        loop.ctx.messages = [Message(role="user", content="test")]

        # 在迭代开始前 stop
        loop.stop()
        result = loop.run_messages()

        assert "终止" in result

    def test_budget_exhaustion_triggers_events_and_hooks(self):
        """预算耗尽时触发事件和钩子。"""
        loop = _make_loop(
            llm_responses=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
                ),
            ],
            max_iterations=1,
        )
        loop.ctx.messages = [Message(role="user", content="do work")]

        # 注册钩子
        hook_calls = []
        class BudgetHook:
            hook_point = HookPoint.ON_BUDGET_EXHAUSTED
            def run(self, ctx):
                hook_calls.append(ctx.iteration)
                return HookResult(action=HookAction.CONTINUE)
        loop.hook_manager.register(BudgetHook())

        events = []
        loop.event_emitter.subscribe(EventType.ON_BUDGET_EXHAUSTED, lambda e: events.append(e))

        result = loop.run_messages()

        assert "迭代" in result
        assert len(events) >= 1
        assert events[0].type == "on_budget_exhausted"
        assert len(hook_calls) >= 1

    def test_stop_in_stream(self):
        """stop() 后 stream_messages 立即返回停止消息。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="should not see")])
        loop.ctx.messages = [Message(role="user", content="test")]

        loop.stop()
        chunks = list(loop.stream_messages())

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "终止" in chunks[0].content

    def test_keyboard_interrupt_in_run(self):
        """KeyboardInterrupt 在 run_messages 中被捕获。"""
        class InterruptingLLM(MockLLM):
            def generate(self, messages, tools=None):
                raise KeyboardInterrupt()

        loop = _make_loop(llm_responses=[])
        loop.llm = InterruptingLLM(responses=[])
        loop.ctx.messages = [Message(role="user", content="hi")]

        result = loop.run_messages()
        assert "终止" in result or "中断" in result

    def test_pre_generate_post_generate_events(self):
        """pre_generate 和 post_generate 事件在循环中触发。"""
        loop = _make_loop(llm_responses=[Message(role="assistant", content="ok")])
        loop.ctx.messages = [Message(role="user", content="test")]

        events = []
        loop.event_emitter.subscribe(EventType.PRE_GENERATE, lambda e: events.append(e.type))
        loop.event_emitter.subscribe(EventType.POST_GENERATE, lambda e: events.append(e.type))

        loop.run_messages()

        assert EventType.PRE_GENERATE in events
        assert EventType.POST_GENERATE in events


class TestLoopEventHooks:
    """验证循环中各生命周期点的钩子和事件触发。"""

    def test_pre_generate_hook_is_called(self):
        loop = _make_loop(llm_responses=[Message(role="assistant", content="ok")])
        loop.ctx.messages = [Message(role="user", content="hi")]

        hook_calls = []
        class GenHook:
            hook_point = HookPoint.PRE_GENERATE
            def run(self, ctx):
                hook_calls.append(("pre_generate", ctx.iteration))
                return HookResult(action=HookAction.CONTINUE)
        loop.hook_manager.register(GenHook())

        loop.run_messages()

        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "pre_generate"

    def test_post_generate_hook_is_called(self):
        loop = _make_loop(llm_responses=[Message(role="assistant", content="ok")])
        loop.ctx.messages = [Message(role="user", content="hi")]

        hook_calls = []
        class PostGenHook:
            hook_point = HookPoint.POST_GENERATE
            def run(self, ctx):
                hook_calls.append(("post_generate", ctx.iteration))
                return HookResult(action=HookAction.CONTINUE)
        loop.hook_manager.register(PostGenHook())

        loop.run_messages()

        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "post_generate"

    def test_pre_tool_post_tool_hooks(self):
        """工具执行时 pre_tool 和 post_tool 钩子被调用。"""
        loop = _make_loop(
            llm_responses=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"x.txt"}'))],
                ),
                Message(role="assistant", content="done"),
            ],
        )
        loop.ctx.messages = [Message(role="user", content="read x.txt")]

        hook_calls = []
        class PreToolHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                hook_calls.append(("pre_tool", ctx.iteration))
                return HookResult(action=HookAction.CONTINUE)
        class PostToolHook:
            hook_point = HookPoint.POST_TOOL
            def run(self, ctx):
                hook_calls.append(("post_tool", ctx.iteration, ctx.tool_call.function.name))
                return HookResult(action=HookAction.CONTINUE)

        loop.hook_manager.register(PreToolHook())
        loop.hook_manager.register(PostToolHook())

        loop.run_messages()

        pre_calls = [c for c in hook_calls if c[0] == "pre_tool"]
        post_calls = [c for c in hook_calls if c[0] == "post_tool"]
        assert len(pre_calls) == 1
        assert len(post_calls) == 1
        assert post_calls[0][2] == "read_file"