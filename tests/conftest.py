"""Agent 测试辅助——创建 Agent 时自动填充必传参数。"""

from __future__ import annotations

from kocor.agent import Agent
from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import EventEmitter
from kocor.hook.hook_manager import HookManager
from kocor.tools.permission import PermissionManager
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
    memory = kwargs.get("memory")  # peek 而非 pop，传入 Agent 时保留
    context = kwargs.pop("context", ContextManager(tools=tm, memory=memory, todo_store=todo))
    pm = kwargs.pop("permission_mgr", PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE))
    hm = kwargs.pop("hook_manager", HookManager())
    ee = kwargs.pop("event_emitter", EventEmitter())
    mi = kwargs.pop("max_iterations", Config.load().max_iterations)
    return Agent(
        llm=llm,
        tool_manager=tm,
        todo_store=todo,
        context=context,
        permission_mgr=pm,
        hook_manager=hm,
        event_emitter=ee,
        max_iterations=mi,
        **kwargs,
    )