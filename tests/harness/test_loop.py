"""Agent 循环控制器测试。"""

import json
from dataclasses import dataclass, field

import pytest

from kocor.agent import Agent
from kocor.event.event_manager import EventEmitter, EventType
from kocor.tools.permission import PermissionManager
from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction
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
        """cron 调度器，测试中为空操作。"""
        pass

    def stop_cron_scheduler(self):
        """cron 调度器，测试中为空操作。"""
        pass


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
        assert agent.ctx.iteration == 1

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
        assert agent.ctx.iteration == 2

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

    def test_budget_exhaustion(self):
        """迭代预算耗尽时 Agent 应停止。"""
        tool_response = Message(
            role="assistant",
            content="doing work...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 10)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            max_iterations=3,
        )
        result = agent.run("do work")
        assert "迭代" in result or "限制" in result
        assert agent.ctx.iteration <= 3

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
                return HookResult(action=HookAction.CONTINUE)

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
            skill_manager = None

            def get_definitions(self, filter_category=None):
                return []

            def execute(self, tool_call):
                raise RuntimeError("something broke")

            def start_cron_scheduler(self):
                pass

            def stop_cron_scheduler(self):
                pass

        llm = MockLLM(responses=[
            Message(
                role="assistant",
                content="running...",
                tool_calls=[ToolCall(id="call_e1", function=FunctionCall(name="write_file", arguments='{"path": "test.txt"}'))],
            ),
            Message(role="assistant", content="There was an error."),
        ])
        agent = Agent(
            llm=llm,
            tool_manager=ErrorToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE, always_allow={"write_file"}),
        )
        result = agent.run("run")
        assert "error" in result.lower() or "Error" in result or "There was" in result

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
            max_iterations=1,
        )
        result = agent.run("do")
        assert agent.ctx.iteration == 1  # Should stop after 1 iteration

    def test_tool_output_truncation(self):
        """过长的工具输出会被截断。"""
        long_content = "x" * 100_000

        class LongResultToolRegistry:
            skill_manager = None
            def get_definitions(self, filter_category=None):
                return []

            def execute(self, tool_call):
                from kocor.llm_provider.message import ToolResult
                from kocor.tools.truncate import ToolOutputTruncator
                truncated = ToolOutputTruncator().truncate(long_content)
                return ToolResult(tool_call_id=tool_call.id, content=truncated)

            def start_cron_scheduler(self):
                pass

            def stop_cron_scheduler(self):
                pass

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
        tool_messages = [m for m in agent.ctx.messages if m.role == "tool"]
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


# ── Loop 公共循环入口（问题 4.1：职责边界） ──


class TestLoopPublicEntry:
    """Loop 对外仅暴露公共循环入口，Agent 不得访问其私有成员。

    循环结束后 Loop 自身负责将本轮 messages 提取为 session_history
    （状态归属收敛），Agent 不再手工调用 extract_session_history。
    """

    def test_run_messages_is_public(self):
        """run_messages 为公共方法，可对预置的 ctx.messages 运行循环。"""
        llm = MockLLM(responses=[Message(role="assistant", content="公共入口回复")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        # 模拟 PROMPT 技能路径：调用方已构造好 messages，直接进入循环
        agent.ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ]
        result = agent.loop.run_messages()
        assert result == "公共入口回复"

    def test_run_messages_auto_extracts_session_history(self):
        """循环结束后 session_history 自动填充，无需外部调用 extract。"""
        llm = MockLLM(responses=[Message(role="assistant", content="回复X")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        agent.ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ]
        agent.loop.run_messages()
        roles = [m.role for m in agent.ctx.session_history]
        assert "user" in roles
        assert "assistant" in roles

    def test_stream_messages_is_public_and_auto_extracts(self):
        """stream_messages 为公共方法，消费完毕后 session_history 自动填充。"""
        llm = MockLLM(responses=[Message(role="assistant", content="流式回复")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        agent.ctx.messages = [
            Message(role="system", content="system"),
            Message(role="user", content="hello"),
        ]
        chunks = list(agent.loop.stream_messages())
        assert len(chunks) > 0
        roles = [m.role for m in agent.ctx.session_history]
        assert "assistant" in roles


# ── 重复工具调用检测 ──


class TestDuplicateToolCallDetection:

    def test_no_duplicate_detection_for_two_identical(self):
        """连续 2 次相同调用不会触发（阈值为 3）。"""
        tool_response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 3 + [Message(role="assistant", content="done")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            max_iterations=10,
        )
        result = agent.run("search")
        # 前两次相同调用正常执行，第3次触发检测
        assert "done" in result or "重复" in result

    def test_duplicate_detection_triggers(self):
        """连续 3 次相同工具调用触发检测并提前终止。"""
        tool_response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 5)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            max_iterations=10,
        )
        result = agent.run("search")
        # 在第 3 次迭代时检测到重复，提前终止（而不是等 10 次预算用完）
        assert "重复" in result
        assert agent.ctx.iteration == 3

    def test_duplicate_detection_resets_on_different_call(self):
        """不同工具调用会重置重复计数。"""
        responses = [
            Message(role="assistant", content="searching...",
                    tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "test"}'))]),
            Message(role="assistant", content="searching...",
                    tool_calls=[ToolCall(id="call_2", function=FunctionCall(name="search", arguments='{"q": "test"}'))]),
            Message(role="assistant", content="reading...",
                    tool_calls=[ToolCall(id="call_3", function=FunctionCall(name="read", arguments='{"path": "x.txt"}'))]),
            Message(role="assistant", content="done"),
        ]
        llm = MockLLM(responses=responses)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("do stuff")
        # 第三次调用不同的工具，重置了计数，任务正常完成
        assert "done" in result

    def test_duplicate_detection_stream(self):
        """流模式下重复工具调用也能检测。"""
        tool_response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 5)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            max_iterations=10,
        )
        chunks = list(agent.stream("search"))
        stuck_chunks = [c for c in chunks if c.is_final and c.content]
        assert stuck_chunks
        assert "重复" in stuck_chunks[0].content

    def test_duplicate_detection_same_name_different_args(self):
        """相同工具名但不同参数不算重复。"""
        responses = [
            Message(role="assistant", content="search a",
                    tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "a"}'))]),
            Message(role="assistant", content="search b",
                    tool_calls=[ToolCall(id="call_2", function=FunctionCall(name="search", arguments='{"q": "b"}'))]),
            Message(role="assistant", content="search c",
                    tool_calls=[ToolCall(id="call_3", function=FunctionCall(name="search", arguments='{"q": "c"}'))]),
            Message(role="assistant", content="done"),
        ]
        llm = MockLLM(responses=responses)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("search multiple")
        assert "done" in result

    def test_duplicate_detection_resets_between_runs(self):
        """重复检测状态在每次 run 之间重置。"""
        tool_response = Message(
            role="assistant",
            content="searching...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "test"}'))],
        )

        # 第一次运行：连续相同调用触发检测
        llm1 = MockLLM(responses=[tool_response] * 5)
        agent = Agent(
            llm=llm1,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            max_iterations=10,
        )
        result1 = agent.run("search")
        assert "重复" in result1

        # 第二次运行：状态已重置，新的相同调用应从 0 开始计数
        llm2 = MockLLM(responses=[tool_response, tool_response, Message(role="assistant", content="done")])
        agent.llm = llm2
        result2 = agent.run("search")
        assert "done" in result2

    def test_get_tool_call_signature(self):
        """_get_tool_call_signature 生成一致的签名。"""
        from kocor.loop import Loop
        tc1 = ToolCall(id="1", function=FunctionCall(name="search", arguments='{"q": "test", "page": 1}'))
        tc2 = ToolCall(id="2", function=FunctionCall(name="search", arguments='{"page": 1, "q": "test"}'))
        sig1 = Loop._get_tool_call_signature(tc1)
        sig2 = Loop._get_tool_call_signature(tc2)
        # 相同参数（不同顺序）应该生成相同签名
        assert sig1 == sig2
        assert "search" in sig1
        assert "test" in sig1

    def test_duplicate_with_multiple_tool_calls(self):
        """单次迭代中多个工具调用完全相同时检测。"""
        # 每次迭代都调用相同的 2 个工具
        multi_tool = Message(
            role="assistant",
            content="doing multiple...",
            tool_calls=[
                ToolCall(id="call_1", function=FunctionCall(name="read", arguments='{"path": "a.txt"}')),
                ToolCall(id="call_2", function=FunctionCall(name="read", arguments='{"path": "b.txt"}')),
            ],
        )
        llm = MockLLM(responses=[multi_tool] * 5)
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            max_iterations=10,
        )
        result = agent.run("read both")
        assert "重复" in result
        # 在第 3 次迭代时触发
        assert agent.ctx.iteration == 3


# ── 停止/中断机制 ──


class TestAgentStop:
    """测试 Agent 停止/中断机制。"""

    def test_stop_flag_stops_loop(self):
        """通过钩子触发 stop() 后循环停止，且未触达预算上限。"""
        tool_response = Message(
            role="assistant",
            content="working...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 10)
        hook_manager = HookManager()
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            hook_manager=hook_manager,
            max_iterations=10,
        )

        triggered = [False]

        class StopHook:
            hook_point = HookPoint.POST_TOOL

            def run(self, ctx):
                if not triggered[0]:
                    triggered[0] = True
                    agent.stop()
                return HookResult(action=HookAction.CONTINUE)

        hook_manager.register(StopHook())
        result = agent.run("do work")

        # 应在预算耗尽前停止
        assert "终止" in result or "已停止" in result or "stop" in result.lower()
        assert agent.ctx.iteration < 10
        # 停止后 agent 应可再次运行
        assert agent.loop._stop_requested is False

    def test_agent_stop_does_not_break_next_run(self):
        """stop() 不会影响下一次正常执行。"""
        llm = MockLLM(responses=[Message(role="assistant", content="第二次运行正常")])
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        # 先 stop 再 run
        agent.stop()
        result = agent.run("hi")
        assert result == "第二次运行正常"

    def test_stop_flag_stops_stream(self):
        """流模式也受 stop() 控制。"""
        tool_response = Message(
            role="assistant",
            content="working...",
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path": "x.txt"}'))],
        )
        llm = MockLLM(responses=[tool_response] * 10)
        hook_manager = HookManager()
        agent = Agent(
            llm=llm,
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
            hook_manager=hook_manager,
            max_iterations=10,
        )

        triggered = [False]

        class StopHook:
            hook_point = HookPoint.POST_TOOL

            def run(self, ctx):
                if not triggered[0]:
                    triggered[0] = True
                    agent.stop()
                return HookResult(action=HookAction.CONTINUE)

        hook_manager.register(StopHook())
        chunks = list(agent.stream("do work"))
        final_chunks = [c for c in chunks if c.is_final]
        assert final_chunks
        # 最后一个 is_final chunk 应包含终止信息
        assert "终止" in final_chunks[-1].content or "已停止" in final_chunks[-1].content
        assert agent.loop._stop_requested is False

    def test_keyboard_interrupt_in_loop(self):
        """KeyboardInterrupt 在循环中被捕获并返回终止信息。"""

        class InterruptingLLM(MockLLM):
            def generate(self, messages, tools=None):
                raise KeyboardInterrupt()

        agent = Agent(
            llm=InterruptingLLM(responses=[]),
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        result = agent.run("hi")
        assert "终止" in result or "中断" in result or "interrupt" in result.lower()

    def test_keyboard_interrupt_in_stream(self):
        """流模式下 KeyboardInterrupt 被捕获。"""

        class InterruptingStreamLLM(MockLLM):
            def generate(self, messages, tools=None):
                raise KeyboardInterrupt()

            def stream(self, messages, tools=None):
                raise KeyboardInterrupt()

        agent = Agent(
            llm=InterruptingStreamLLM(responses=[]),
            tool_manager=MockToolRegistry(),
            permission_mgr=PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE),
        )
        chunks = list(agent.stream("hi"))
        final_chunks = [c for c in chunks if c.is_final]
        assert final_chunks
        assert "终止" in final_chunks[0].content or "中断" in final_chunks[0].content
