"""Agent 测试辅助——创建 Agent 时自动填充必传参数。"""

from __future__ import annotations

from unittest.mock import MagicMock

from kocor.agent import Agent
from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import EventEmitter
from kocor.hook.hook_manager import HookManager
from kocor.tools.tool_manager import ToolManager
from kocor.tools.toolsets.todo_tool import TodoStore


def make_agent(llm, **kwargs):
    """创建测试用 Agent，自动填充必传参数。

    用法：
        from tests.conftest import make_agent
        agent = make_agent(llm=llm, ...)

    支持覆盖默认值：
        agent = make_agent(llm=llm, tool_manager=my_tm, session_manager=my_sm)
    """
    todo = kwargs.pop("todo_store", TodoStore())
    tm = kwargs.pop("tool_manager", ToolManager())

    # 确保 tool_manager 有 permission_mgr（MagicMock 默认无此属性）
    if not hasattr(tm, 'permission_mgr'):
        pm = MagicMock()
        pm.check.return_value = True
        tm.permission_mgr = pm
    else:
        # 默认 permissive 策略，避免阻塞 stdin
        tm.permission_mgr.policy = "permissive"

    # 允许测试注入自定义 permission_mgr
    custom_pm = kwargs.pop("permission_mgr", None)
    if custom_pm is not None:
        tm.permission_mgr = custom_pm
    memory_store = kwargs.get("memory_store")  # peek 而非 pop，传入 Agent 时保留
    # 注入共享 store（与 AgentBuilder.build() 行为一致）
    tm.todo_store = todo
    if memory_store is not None:
        tm.memory_store = memory_store
    context = kwargs.pop("context", ContextManager(tools=tm, memory=memory_store, todo_store=todo))
    hm = kwargs.pop("hook_manager", HookManager())
    ee = kwargs.pop("event_emitter", EventEmitter())
    mi = kwargs.pop("max_iterations", Config.load().max_iterations)
    return Agent(
        llm=llm,
        tool_manager=tm,
        todo_store=todo,
        context=context,
        hook_manager=hm,
        event_emitter=ee,
        max_iterations=mi,
        **kwargs,
    )