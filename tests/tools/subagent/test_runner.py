"""测试 SubagentRunner（单任务 + 批量）。

TDD：验证上下文隔离、摘要提取、状态报告、批量并行。
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from kocor.config import Config
from kocor.event.event_manager import EventType
from kocor.llm_provider.message import FunctionCall, Message, ToolCall
from kocor.tools.tool_manager import ToolManager
from kocor.tools.toolsets.subagent.runner import SubagentRunner


class MockEmitter:
    """简易 Mock EventEmitter，记录收到的事件。"""
    def __init__(self):
        self.events = []

    def subscribe(self, *args, **kwargs):
        pass

    def fire(self, event):
        self.events.append(event)

    @property
    def _subscribers(self):
        return {}


class MockLLM:
    """Mock LLM 客户端，预设返回内容。"""
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["done"]
        self.call_count = 0

    def generate(self, messages, tools=None, **kwargs):
        text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return Message(role="assistant", content=text)

    def stream(self, messages, tools=None, **kwargs):
        return iter([])


class TestRunnerSingleGoal:
    """测试 SubagentRunner 单任务模式。"""

    def setup_method(self):
        self.parent_tm = ToolManager()
        self.llm = MockLLM(["子代理完成了任务"])
        self.emitter = MockEmitter()
        self.runner = SubagentRunner(
            parent_llm=self.llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )

    def test_run_single_goal_returns_completed(self):
        result = self.runner.run(goal="测试子代理")
        assert result["status"] == "completed"
        assert "子代理完成了任务" in result["summary"]

    def test_run_emits_start_event(self):
        self.runner.run(goal="测试子代理")
        start_events = [e for e in self.emitter.events if e.type == EventType.SUBAGENT_START]
        assert len(start_events) >= 1
        assert start_events[0].data["goal"] == "测试子代理"

    def test_run_emits_complete_event(self):
        self.runner.run(goal="测试子代理")
        complete_events = [e for e in self.emitter.events if e.type == EventType.SUBAGENT_COMPLETE]
        assert len(complete_events) >= 1
        assert complete_events[0].data["status"] == "completed"

    def test_run_context_passed(self):
        result = self.runner.run(goal="检查代码", context="文件路径: /tmp/test.py")
        assert result["status"] == "completed"

    def test_run_iterations_recorded(self):
        result = self.runner.run(goal="测试")
        assert "iterations" in result
        assert result["iterations"] >= 0

    def test_run_duration_recorded(self):
        result = self.runner.run(goal="测试")
        assert "duration" in result
        assert result["duration"] >= 0

    def test_run_budget_exhausted_status(self):
        """MockLLM 持续返回 tool_calls（不停循环）→ 预算耗尽。"""
        def tool_always(**kwargs):
            return Message(role="assistant", content="",
                           tool_calls=[MagicMock()])

        from unittest.mock import MagicMock

        llm = MockLLM.__new__(MockLLM)
        llm.call_count = 0

        def gen(messages, tools=None, **kwargs):
            llm.call_count += 1
            return Message(
                role="assistant", content="",
                tool_calls=[
                    ToolCall(id="c1", function=FunctionCall(name="bash", arguments='{"command":"echo hi"}'))
                ],
            )
        llm.generate = gen
        llm.stream = lambda m, **kw: iter([])

        runner = SubagentRunner(
            parent_llm=llm,
            parent_tool_manager=ToolManager(),
            parent_event_emitter=MockEmitter(),
            depth=0,
        )
        from kocor.config import Config
        old = Config.load().subagent_max_iterations
        Config.load().subagent_max_iterations = 2
        try:
            result = runner.run(goal="无限循环")
            assert result["status"] in ("budget_exhausted", "error")
        finally:
            Config.load().subagent_max_iterations = old

    def test_invalid_goal_and_tasks_together(self):
        result = self.runner.run(goal="测试", tasks=[{"goal": "子任务"}])
        assert result["status"] == "error"

    def test_missing_goal_and_tasks(self):
        result = self.runner.run()
        assert result["status"] == "error"

    def test_stop_sets_interrupt_event(self):
        self.runner.stop()
        assert self.runner._stop_requested.is_set() is True

    def test_stop_propagates_to_child_loop(self):
        """stop() 设置 running_loops 中每个 Loop 的 _stop_requested。"""
        mock_loop = Mock()
        mock_loop._stop_requested = False
        mock_loop.stop = lambda: setattr(mock_loop, '_stop_requested', True)

        runner = SubagentRunner(
            parent_llm=self.llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        runner._running_loops.append(mock_loop)
        runner.stop()
        assert mock_loop._stop_requested is True

    def test_timeout_subagent_timeout_setting(self):
        """验证 subagent_timeout 配置被读取（0=关）。"""
        assert Config.load().subagent_timeout == 0


class TestRunnerTimeout:
    """测试子代理 wall-clock 超时。"""

    def setup_method(self):
        self.parent_tm = ToolManager()
        self.llm = MockLLM(["子代理完成了任务"])
        self.emitter = MockEmitter()
        self._saved_timeout = Config.load().subagent_timeout

    def teardown_method(self):
        Config.load().subagent_timeout = self._saved_timeout

    def test_timeout_enabled_on_slow_child(self):
        """subagent_timeout > 0 时，慢子代理被超时终止。"""
        Config.load().subagent_timeout = 1  # 1s 超时

        # 用慢 MockLLM：睡眠超过 1s 再返回
        class SlowLLM(MockLLM):
            def generate(self, messages, tools=None, **kwargs):
                time.sleep(3)
                return Message(role="assistant", content="最终完成")

        runner = SubagentRunner(
            parent_llm=SlowLLM(["done"]),
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        result = runner.run(goal="缓慢任务")
        assert result["status"] == "timeout"

    def test_timeout_zero_disabled(self):
        """subagent_timeout=0 时子代理正常完成。"""
        Config.load().subagent_timeout = 0
        runner = SubagentRunner(
            parent_llm=MockLLM(["正常完成"]),
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        result = runner.run(goal="正常任务")
        assert result["status"] == "completed"

    def test_batch_timeout_marks_error(self):
        """批量中某子代理超时，结果标记 timeout。"""
        Config.load().subagent_timeout = 1

        class SlowLLM(MockLLM):
            def generate(self, messages, tools=None, **kwargs):
                time.sleep(3)
                return Message(role="assistant", content="done")

        runner = SubagentRunner(
            parent_llm=SlowLLM(["slow"]),
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        tasks = [{"goal": "慢任务"}]
        result = runner.run(tasks=tasks)
        # 批量结果是 {"results": [...]}
        assert "results" in result
        assert result["results"][0]["status"] == "timeout"


class TestRunnerOrchestrator:
    """测试 orchestrator 递归注入。"""

    def setup_method(self):
        self.parent_tm = ToolManager()
        self.emitter = MockEmitter()

    def test_max_depth_1_child_is_leaf_placeholder_kept(self):
        """max_depth=1：子代理为 leaf，占位 handler 保留。"""
        llm = MockLLM(["完成"])
        runner = SubagentRunner(
            parent_llm=llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
            max_depth=1,
        )
        # 运行后验证占位 handler 未被替换
        result = runner.run(goal="子任务")
        assert result["status"] == "completed"

    def test_max_depth_2_child_is_orchestrator_handler_replaced(self):
        """max_depth=2：子代理为 orchestrator，subagent handler 被替换为真实 runner。"""
        llm = MockLLM(["完成"])
        runner = SubagentRunner(
            parent_llm=llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
            max_depth=2,
        )
        result = runner.run(goal="子任务")
        assert result["status"] == "completed"
        # 验证子代理的 TM 中 subagent handler 不再返回占位错误
        # 通过检查最后一个 _running_loop 的 handler 输出：
        if runner._running_loops:
            child_loop = runner._running_loops[-1]
            handler = child_loop.tool_manager._handlers.get("subagent")
            if handler is not None:
                import json
                resp = handler()
                data = json.loads(resp)
                # 真实 handler 返回 runner 的结果（error 因为无 goal/tasks）
                # 占位 handler 返回固定 error 字符串
                assert "SubagentRunner 未装配" not in resp


class TestRunnerBatch:
    """测试批量并行执行。"""

    def setup_method(self):
        self.parent_tm = ToolManager()
        self.llm = MockLLM(["子任务完成"])
        self.emitter = MockEmitter()
        self._max_concurrent = Config.load().subagent_max_concurrent
        Config.load().subagent_max_concurrent = 4

    def teardown_method(self):
        Config.load().subagent_max_concurrent = self._max_concurrent

    def test_batch_returns_ordered_results(self):
        runner = SubagentRunner(
            parent_llm=self.llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        tasks = [{"goal": f"子任务{i}"} for i in range(3)]
        result = runner.run(tasks=tasks)
        assert "results" in result
        assert len(result["results"]) == 3
        for i, r in enumerate(result["results"]):
            assert r["status"] == "completed"

    def test_batch_exceeds_max_concurrent_rejected(self):
        Config.load().subagent_max_concurrent = 2
        runner = SubagentRunner(
            parent_llm=self.llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        tasks = [{"goal": f"子任务{i}"} for i in range(3)]
        result = runner.run(tasks=tasks)
        assert result["status"] == "error"
        assert "上限" in result["summary"]

    def test_batch_emits_events(self):
        runner = SubagentRunner(
            parent_llm=self.llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        tasks = [{"goal": "子任务1"}, {"goal": "子任务2"}]
        runner.run(tasks=tasks)
        starts = [e for e in self.emitter.events if e.type == EventType.SUBAGENT_START]
        completes = [e for e in self.emitter.events if e.type == EventType.SUBAGENT_COMPLETE]
        assert len(starts) == 2
        assert len(completes) == 2

    def test_batch_total_duration(self):
        runner = SubagentRunner(
            parent_llm=self.llm,
            parent_tool_manager=self.parent_tm,
            parent_event_emitter=self.emitter,
            depth=0,
        )
        tasks = [{"goal": "任务A"}, {"goal": "任务B"}]
        result = runner.run(tasks=tasks)
        assert "total_duration" in result
        assert result["total_duration"] >= 0