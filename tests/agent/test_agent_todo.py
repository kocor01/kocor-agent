"""测试 Agent 装配 TodoStore 及 hydrate 钩子。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kocor.agent import Agent
from kocor.llm_provider.message import FunctionCall, Message, ToolCall
from tests.agent.test_agent import FakeLLMClient


def _todo_result(todos: list[dict]) -> str:
    return json.dumps({"todos": todos, "summary": {}}, ensure_ascii=False)


class TestToolManagerTodoRegistration:
    """测试 ToolManager 注册 todo 工具。"""

    def test_register_builtin_tools_includes_todo(self):
        from kocor.tools.tool_manager import ToolManager
        tm = ToolManager()
        tm.register_builtin_tools()
        assert "todo" in tm._tools
        assert tm.todo_store is None  # 未注入 Agent 前为 None

    def test_todo_handler_uses_injected_store(self):
        """注册后 handler 闭包应使用注入的 todo_store。"""
        from kocor.tools.tool_manager import ToolManager
        from kocor.tools.toolsets.todo_tool import TodoStore
        tm = ToolManager()
        tm.register_builtin_tools()
        tm.todo_store = TodoStore()

        handler = tm._handlers["todo"]
        result = handler(todos=[{"id": "1", "content": "via tm", "status": "pending"}])
        data = json.loads(result)
        assert data["summary"]["total"] == 1
        assert tm.todo_store.read()[0]["content"] == "via tm"


class TestAgentTodoWiring:
    """测试 Agent 初始化时装配 TodoStore 并注册 todo 工具。"""

    def test_init_creates_todo_store(self):
        llm = FakeLLMClient([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm)
        assert agent._todo_store is not None
        assert agent._todo_store.has_items() is False

    def test_todo_store_injected_to_tool_manager(self):
        llm = FakeLLMClient([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm)
        assert agent.tool_manager.todo_store is agent._todo_store

    def test_todo_store_injected_to_context_manager(self):
        llm = FakeLLMClient([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm)
        assert agent.ctx.todo_store is agent._todo_store


class TestAgentTodoHydrate:
    """测试会话恢复时的 hydrate 钩子。"""

    def test_hydrate_restores_from_history_when_empty(self):
        """store 为空时，hydrate 从历史回填最后一次 todo 结果。"""
        llm = FakeLLMClient([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm)

        history = [
            Message(role="user", content="start"),
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(name="todo", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c1", content=_todo_result([
                {"id": "1", "content": "task from history", "status": "in_progress"},
            ])),
        ]
        agent._hydrate_todo_store(history)
        assert agent._todo_store.has_items() is True
        assert agent._todo_store.read()[0]["content"] == "task from history"

    def test_hydrate_skipped_when_store_nonempty(self):
        """store 非空时，hydrate 不覆盖实时状态。"""
        llm = FakeLLMClient([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm)
        agent._todo_store.write([{"id": "9", "content": "live task", "status": "pending"}])

        history = [
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(name="todo", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c1", content=_todo_result([
                {"id": "1", "content": "stale from history", "status": "pending"},
            ])),
        ]
        agent._hydrate_todo_store(history)
        items = agent._todo_store.read()
        assert len(items) == 1
        assert items[0]["content"] == "live task"  # 未被历史覆盖
