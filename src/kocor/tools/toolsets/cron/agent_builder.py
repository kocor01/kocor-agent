"""cron worker 子进程内装配独立 Agent + CronScheduler。

职责：在子进程启动时统一组装 cron 所需的：
  - 一个完全独立的 Agent（无 cronjob 工具、无子进程 worker）
  - 一个持有该 Agent 引用的 CronScheduler（子进程内 tick 委托）

调用方（cron_worker.main）无需关心组装细节。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_cron_agent() -> tuple:
    """构建 cron worker 专属 Agent + CronScheduler。

    返回 (Agent, CronScheduler) —— 子进程主循环只需分别启动它们。

    装配细则：
    - ToolManager 以 include_cron=False 注册，不添加 cronjob 工具
      （防递归调用）且不创建 CronWorkerProcess（避免递归 spawn）。
    - Agent 不设 memory_store：cron 自主作业不需要跨会话人类记忆，
      且避免与主进程并发读写 memory 文件。
    - CronScheduler 持有 Agent 引用：tick 到期 prompt 作业时
      委托 Agent.run_prompt 执行 ReAct 循环。
    """
    from kocor.config import Config

    # cron 自主作业不需要跨会话人类记忆。禁用 memory 同时避免与主
    # 进程并发读写 memory 文件（进程隔离要求）。
    Config.load().memory_enabled = False

    from kocor.agent import Agent
    from kocor.context.context_manager import ContextManager
    from kocor.event.event_manager import EventEmitter
    from kocor.hook.hook_manager import HookManager
    from kocor.llm_provider.llm_factory import LlmFactory
    from kocor.tools.tool_manager import ToolManager
    from kocor.tools.toolsets.cron.scheduler import CronScheduler
    from kocor.tools.toolsets.todo_tool import TodoStore

    # 独立 Agent（无 cronjob 工具，无 worker 子进程）
    llm = LlmFactory.create()
    tool_manager = ToolManager()
    tool_manager.permission_mgr.policy = "permissive"
    tool_manager.register_builtin_tools(include_cron=False)
    todo_store = TodoStore()
    context = ContextManager(tools=tool_manager, todo_store=todo_store)
    cfg = Config.load()
    agent = Agent(
        llm=llm,
        tool_manager=tool_manager,
        todo_store=todo_store,
        context=context,
        hook_manager=HookManager(),
        event_emitter=EventEmitter(),
        max_iterations=cfg.max_iterations,
    )

    # 子进程内 tick 调度器，持有 agent 引用
    scheduler = CronScheduler(agent=agent)

    return agent, scheduler