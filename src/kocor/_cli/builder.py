"""Agent 装配工厂。

将 Agent 及其依赖组件的装配过程封装为可链式调用的 Builder，
消除 CLI main() 中 100+ 行的组件组装代码。
"""

from __future__ import annotations

import atexit

from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_factory import LlmFactory
from kocor.tools.tool_manager import ToolManager
from kocor.tools.permission import PermissionManager
from kocor.hook.hook_manager import HookManager
from kocor.event.event_manager import EventEmitter
from kocor.event.event_subscribe import EventSubscribe
from kocor.logger import Logger


class AgentBuilder:
    """负责装配 Agent 及其所有依赖组件的工厂类。

    职责链：基础组件 → 插件系统 → 组装 → 生命周期管理。

    用法:
        agent = (
            AgentBuilder()
            .build_llm()
            .build_tools()
            .build_permission()
            .build_hooks(logger)
            .build_session()
            .build()
        )
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()
        self.tool_manager = ToolManager()
        self.event_emitter = EventEmitter()
        self.llm = None
        self.hook_manager = HookManager()
        self.permission_mgr = None
        self.session_manager = None

    def build_llm(self) -> AgentBuilder:
        """创建 LLM 客户端。"""
        self.llm = LlmFactory.create()
        return self

    def build_subagent(self) -> AgentBuilder:
        """创建 SubagentRunner（可选）。

        必须在 build_llm 之后、build_tools 之前调用。
        """
        if self.config.subagent_enabled:
            from kocor.tools.toolsets.subagent.runner import SubagentRunner

            runner = SubagentRunner(
                parent_llm=self.llm,
                parent_tool_manager=self.tool_manager,
                parent_event_emitter=self.event_emitter,
                depth=0,
            )
            self.tool_manager._subagent_runner = runner
        return self

    def build_tools(self) -> AgentBuilder:
        """注册所有工具到 ToolManager。"""
        self.tool_manager.register_all(
            include_subagent=self.config.subagent_enabled,
        )
        return self

    def build_permission(self) -> AgentBuilder:
        """创建 PermissionManager。"""
        self.permission_mgr = PermissionManager(
            policy=self.config.permission_policy,
            tool_manager=self.tool_manager,
        )
        return self

    def build_hooks(self, logger: Logger) -> AgentBuilder:
        """注册钩子和事件订阅。"""
        self.hook_manager.register_all(logger=logger)
        EventSubscribe(self.event_emitter).subscribe_all(logger=logger)
        return self

    def build_session(self) -> AgentBuilder:
        """创建会话管理器（可选）。"""
        if self.config.session_enabled:
            from kocor.session import SessionManager, SessionResetPolicy, SessionStore

            db_path = self.config.session_db_path
            session_name = self.config.session_name or None
            store = SessionStore(db_path=db_path)
            policy = SessionResetPolicy(mode="none")
            self.session_manager = SessionManager(
                store=store,
                policy=policy,
                profile=session_name,
            )
        return self

    def build(self) -> Agent:
        """组装并返回 Agent 实例。

        默认流程：build_llm → build_subagent → build_tools → build_permission → build。
        也可手动调用各 build_* 方法后最后调用 build()。
        """
        agent = Agent(
            llm=self.llm,
            tool_manager=self.tool_manager,
            permission_mgr=self.permission_mgr,
            hook_manager=self.hook_manager,
            event_emitter=self.event_emitter,
            max_iterations=self.config.max_iterations,
            session_manager=self.session_manager,
        )

        # 注册进程退出清理
        atexit.register(self.tool_manager.stop_cron_scheduler)
        return agent