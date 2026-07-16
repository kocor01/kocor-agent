"""测试 SubagentTool handler。

TDD：验证 handler 通过 runner 执行并返回 JSON 结构。
"""

from __future__ import annotations

import json
from unittest.mock import Mock

from kocor.tools.toolsets.subagent.tool import SubagentTool


class TestSubagentToolHandler:
    """测试 SubagentTool.handler 静态方法。"""

    def test_handler_calls_runner_with_goal(self):
        runner = Mock()
        runner.run = Mock(return_value={"status": "completed", "summary": "done"})
        result = SubagentTool.handler(runner=runner, goal="测试任务")
        data = json.loads(result)
        assert data["status"] == "completed"
        runner.run.assert_called_once_with(goal="测试任务", context=None, tasks=None)

    def test_handler_calls_runner_with_context(self):
        runner = Mock()
        runner.run = Mock(return_value={"status": "completed", "summary": "done"})
        SubagentTool.handler(runner=runner, goal="测试", context="背景信息")
        runner.run.assert_called_once_with(goal="测试", context="背景信息", tasks=None)

    def test_handler_calls_runner_with_tasks(self):
        runner = Mock()
        runner.run = Mock(return_value={"results": [{"status": "completed"}]})
        tasks = [{"goal": "子任务1"}, {"goal": "子任务2"}]
        SubagentTool.handler(runner=runner, tasks=tasks)
        runner.run.assert_called_once_with(goal=None, context=None, tasks=tasks)

    def test_handler_no_runner_returns_error(self):
        result = SubagentTool.handler(goal="测试")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "未装配" in data["summary"]

    def test_handler_returns_json(self):
        runner = Mock()
        runner.run = Mock(return_value={"status": "completed", "summary": "OK"})
        result = SubagentTool.handler(runner=runner, goal="测试")
        # 应返回有效的 JSON 字符串
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_handler_extra_kwargs_ignored(self):
        """handler 接受 **kwargs 以兼容 ToolManager 的调用方式。"""
        runner = Mock()
        runner.run = Mock(return_value={"status": "completed", "summary": "OK"})
        result = SubagentTool.handler(runner=runner, goal="测试", unexpected_arg="val")
        data = json.loads(result)
        assert data["status"] == "completed"