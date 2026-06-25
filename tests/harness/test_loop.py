"""Agent 循环控制器测试。"""

import json
from dataclasses import dataclass, field

import pytest

from kocor.agent import Agent
from kocor.harness.loop import ToolCallRecord
from kocor.harness.budget import IterationBudget
from kocor.harness.events import EventEmitter, EventType
from kocor.tools.permission import PermissionManager
from kocor.hook.base import HookPoint, HookContext, HookResult
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.message import Message, ToolCall, FunctionCall, StreamChunk


# ── 辅助方法 ──


class MockLLM:
    """模拟 LLM，返回预设的响应。"""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0
        self.messages_history = []

    def generate(self, messages, tools=None):
        self.messages_history.append(messages[:])
        response = self.responses[self.call_count] if self.call_count < len(self.responses) else Message(role="assistant", content="done")
        self.call_count += 1
        return response

    @property
    def provider(self) -> str:
        return "fake"

    def stream(self, messages, tools=None):
        """模拟流式 - 将内容分块产出，然后产出最终块。"""
        response = self.generate(messages, tools)
        if response.content:
            yield StreamChunk(content=response.content, is_final=False)
        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamChunk(tool_calls=[tc], is_final=False)
        yield StreamChunk(is_final=True)


class MockToolRegistry:
    """模拟工具注册器。"""

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


# ── 测试 ──


class TestAgentLoop:
    def test_simple_response_no_tools(self):
        """LLM 直接回答时，Agent 应立即返回。"""
        llm = MockLLM(responses=[
            Message(role="assistant", content="Hello! How can I help?"),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("hi")
        assert result == "Hello! How can I help?"
        assert agent._iteration == 1

    def test_single_tool_call(self):
        """Agent 执行工具并返回结果。"""
        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="Let me read that file.",
                tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'))],
            ),
            Message(role="assistant", content="The file contains: hello"),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("read test.txt")
        assert "hello" in result
        assert agent._iteration == 2

    def test_multiple_tool_calls_one_iteration(self):
        """Agent 在一次 LLM 响应中处理多个工具调用。"""
        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="Let me do both.",
                tool_calls=[
                    ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}')),
                    ToolCall(id="call_2", function=FunctionCall(name="read_file", arguments='{"path": "b.txt"}')),
                ],
            ),
            Message(role="assistant", content="Both files read."),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("read both")
        assert "Both files read." in result
        assert len(agent.get_tool_history()) == 2

    def test_permission_denied(self):
        """权限管理器拒绝的工具调用会返回错误给 LLM。"""
        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="Let me write that.",
                tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="write_file", arguments='{"path": "test.txt", "content": "data"}'))],
            ),
            Message(role="assistant", content="I see it was denied."),
        ])
        pm = PermissionManager(policy=PermissionManager.POLICY_STRICT, always_ask={"write_file"})
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=pm,
        )
        result = agent.run("write file")
        assert "denied" in result.lower() or "I see" in result
        # 该工具不应被执行
        assert len(agent._tool_history) == 1
        assert agent._tool_history[0].permission == "denied"

    def test_budget_exhaustion(self):
        """迭代预算耗尽时 Agent 应停止。"""
        tool_response = Message(
            role="assistant",
            content="doing work...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 10)
        budget = IterationBudget(iterations_limit=3)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            budget=budget,
        )
        result = agent.run("do work")
        assert "迭代" in result or "限制" in result
        assert agent._iteration <= 3

    def test_tool_history_tracking(self):
        """每次工具执行都会创建 ToolCallRecord。"""
        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="reading...",
                tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'))],
            ),
            Message(role="assistant", content="done"),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        agent.run("read")
        history = agent.get_tool_history()
        assert len(history) == 1
        record = history[0]
        assert record.tool_name == "read_file"
        assert record.permission in ("auto", "confirm")
        assert record.duration_ms >= 0
        assert record.result_token_count > 0

    def test_pre_generate_event(self):
        """事件发射器触发 pre_generate 事件。"""
        emitter = EventEmitter()
        events = []
        emitter.subscribe(EventType.PRE_GENERATE, lambda e: events.append(e))

        llm = MockLLM(responses=[Message(role="assistant", content="ok")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            event_emitter=emitter,
        )
        agent.run("test")
        assert any(e.type == "pre_generate" for e in events)

    def test_hook_manager_integration(self):
        """HookManager 在生命周期点被调用。"""
        hook_manager = HookManager()
        hook_calls = []

        class TestHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                hook_calls.append(("pre_tool", ctx.iteration))
                return HookResult(action="continue")

        hook_manager.register(TestHook())
        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="reading...",
                tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
            ),
            Message(role="assistant", content="done"),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            hook_manager=hook_manager,
        )
        agent.run("read")
        assert len(hook_calls) >= 1
        assert hook_calls[0][0] == "pre_tool"

    def test_tool_error_handling(self):
        """工具执行错误被优雅捕获。"""
        class ErrorToolRegistry:
            def get_definitions(self, filter_category=None):
                return []

            def execute(self, tool_call):
                raise RuntimeError("something broke")

        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="running...",
                tool_calls=[ToolCall(id="call_e1", function=FunctionCall(name="run_python", arguments='{"code": "bad"}'))],
            ),
            Message(role="assistant", content="There was an error."),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=ErrorToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE, always_allow={"run_python"}),
        )
        result = agent.run("run")
        assert "error" in result.lower() or "Error" in result or "There was" in result
        assert agent._tool_history[0].error is not None

    def test_stream_basic(self):
        """流模式应产出数据块。"""
        llm = MockLLM(responses=[Message(role="assistant", content="streaming result")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        chunks = list(agent.stream("test"))
        assert len(chunks) > 0
        contents = [c.content for c in chunks if c.content]
        assert any("streaming" in c for c in contents)

    def test_stream_with_tool_call(self):
        """流模式正确处理工具调用。"""
        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="Let me check...",
                tool_calls=[ToolCall(id="call_s1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
            ),
            Message(role="assistant", content="Final answer"),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        chunks = list(agent.stream("test"))
        assert len(chunks) > 0
        final_chunks = [c for c in chunks if "Final" in (c.content or "")]
        assert final_chunks

    def test_max_iterations_config(self):
        """Agent 遵循预算中的最大迭代次数。"""
        budget = IterationBudget(iterations_limit=1)
        tool_response = Message(
            role="assistant",
            content="working...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 5)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            budget=budget,
        )
        result = agent.run("do")
        assert agent._iteration == 1  # Should stop after 1 iteration

    def test_tool_output_truncation(self):
        """过长的工具输出会被截断。"""
        long_content = "x" * 100_000

        class LongResultToolRegistry:
            def get_definitions(self, filter_category=None):
                return []

            def execute(self, tool_call):
                from kocor.llm_provider.message import ToolResult
                return ToolResult(tool_call_id=tool_call.id, content=long_content)

        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="reading...",
                tool_calls=[ToolCall(id="call_t1", function=FunctionCall(name="read_file", arguments='{"path": "big.txt"}'))],
            ),
            Message(role="assistant", content="done reading big file"),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=LongResultToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        agent.run("read big file")
        tool_messages = [m for m in agent._messages if m.role == "tool"]
        if tool_messages:
            assert len(tool_messages[-1].content) < len(long_content)

    def test_run_with_empty_input(self):
        """空输入也能正常运行。"""
        llm = MockLLM(responses=[Message(role="assistant", content="")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("")
        assert result is not None


class TestToolCallRecord:
    def test_create_record(self):
        record = ToolCallRecord(
            iteration=1,
            tool_name="read_file",
            arguments={"path": "test.txt"},
            result_summary="file content",
            result_token_count=100,
            duration_ms=50.0,
            permission="auto",
        )
        assert record.tool_name == "read_file"
        assert record.iteration == 1
        assert record.error is None

    def test_create_record_with_error(self):
        record = ToolCallRecord(
            iteration=1,
            tool_name="run_python",
            arguments={},
            result_summary="Error!",
            result_token_count=0,
            duration_ms=10.0,
            permission="auto",
            error="RuntimeError",
        )
        assert record.error == "RuntimeError"
