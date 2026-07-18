"""Agent 装配工厂。

将 Agent 及其依赖组件的装配过程封装为 Builder，
消除 CLI main() 中 100+ 行的组件组装代码。
"""

import atexit

from kocor.agent import Agent
from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import EventEmitter
from kocor.event.event_subscribe import EventSubscribe
from kocor.event.subscribes.metrics import MetricsCollector
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.llm_factory import LlmFactory
from kocor.logger import Logger
from kocor.memory.reviewer import BackgroundReviewer
from kocor.memory.store import MemoryStore
from kocor.session.manager import SessionManager
from kocor.tools.tool_manager import ToolManager
from kocor.tools.toolsets.todo_tool import TodoStore


class AgentBuilder:
    """负责装配 Agent 及其所有依赖组件的工厂类。

    职责链：基础组件 → 插件系统 → 组装 → 生命周期管理。

    用法:
        agent = AgentBuilder().build(logger=logger)
    """

    def __init__(self):
        # 所有组件由对应 _init_* 方法按顺序创建，构造时仅声明为 None
        self.tool_manager: ToolManager | None = None
        self.event_emitter: EventEmitter | None = None
        self.hook_manager: HookManager | None = None
        self.llm: LLMClient | None = None
        self.session_manager: SessionManager | None = None
        self._metrics: MetricsCollector | None = None
        self._memory: MemoryStore | None = None
        self._background_reviewer: BackgroundReviewer | None = None
        self._todo_store: TodoStore | None = None
        self.context: ContextManager | None = None

    def _init_llm(self) -> None:
        """创建 LLM 客户端。"""
        self.llm = LlmFactory.create()

    def _init_memory(self) -> None:
        """创建 MemoryStore 和 BackgroundReviewer（可选）。"""
        if Config.load().memory_enabled:
            memory_dir = Config.load().memory_dir or None
            if memory_dir:
                self._memory = MemoryStore(
                    memory_dir=memory_dir,
                    memory_limit=Config.load().memory_char_limit,
                    user_limit=Config.load().user_char_limit,
                    user_enabled=Config.load().user_profile_enabled,
                )
                self._memory.load_from_disk()
                if self.tool_manager is None:
                    self.tool_manager = ToolManager()
                self.tool_manager.memory_store = self._memory
                self._background_reviewer = BackgroundReviewer(
                    llm=self.llm, store=self._memory
                )

    def _init_subagent(self) -> None:
        """创建 SubagentRunner（可选）。

        必须在 _init_llm 之后调用。
        """
        if Config.load().subagent_enabled:
            from kocor.tools.toolsets.subagent.runner import SubagentRunner

            if self.tool_manager is None:
                self.tool_manager = ToolManager()
            if self.event_emitter is None:
                self.event_emitter = EventEmitter()
            runner = SubagentRunner(
                parent_llm=self.llm,
                parent_tool_manager=self.tool_manager,
                parent_event_emitter=self.event_emitter,
                depth=0,
            )
            self.tool_manager._subagent_runner = runner

    def _init_todo_store(self) -> None:
        """创建 TodoStore 并注入 tool_manager。"""
        if self.tool_manager is None:
            self.tool_manager = ToolManager()
        self._todo_store = TodoStore()
        self.tool_manager.todo_store = self._todo_store

    def _init_tool_manager(self) -> None:
        """创建 ToolManager 并注册所有工具。"""
        if self.tool_manager is None:
            self.tool_manager = ToolManager()
        self.tool_manager.register_all(
            include_subagent=Config.load().subagent_enabled,
        )

    def _init_hook_manager(self, logger: Logger) -> None:
        """创建 HookManager、EventEmitter 并注册钩子和事件订阅。"""
        if self.hook_manager is None:
            self.hook_manager = HookManager()
        if self.event_emitter is None:
            self.event_emitter = EventEmitter()
        self.hook_manager.register_all(logger=logger)
        self._metrics = MetricsCollector()
        EventSubscribe(self.event_emitter).subscribe_all(logger=logger, metrics=self._metrics)

    def _init_session_manager(self) -> None:
        """创建会话管理器（可选）。"""
        if Config.load().session_enabled:
            from kocor.session import SessionResetPolicy, SessionStore

            db_path = Config.load().session_db_path
            session_name = Config.load().session_name or None
            store = SessionStore(db_path=db_path)
            policy = SessionResetPolicy(mode="none")
            self.session_manager = SessionManager(
                store=store,
                policy=policy,
                profile=session_name,
            )

    def _init_context(self) -> None:
        """创建 ContextManager。

        需要 tool_manager、_memory、_todo_store 均已就绪。
        """
        self.context = ContextManager(
            tools=self.tool_manager,
            memory=self._memory,
            todo_store=self._todo_store,
        )

    def build(self, logger: Logger) -> Agent:
        """组装并返回 Agent 实例。

        按依赖顺序依次调用各内部构建方法：
        LLM → 记忆 → Subagent → Todo → 工具 → 权限 → 钩子 → 会话 → 上下文 → 组装
        """
        self._init_llm()
        self._init_memory()
        self._init_subagent()
        self._init_todo_store()
        self._init_tool_manager()
        self._init_hook_manager(logger)
        self._init_session_manager()
        self._init_context()
        agent = Agent(
            llm=self.llm,
            tool_manager=self.tool_manager,
            todo_store=self._todo_store,
            context=self.context,
            hook_manager=self.hook_manager,
            event_emitter=self.event_emitter,
            max_iterations=Config.load().max_iterations,
            session_manager=self.session_manager,
            memory=self._memory,
            background_reviewer=self._background_reviewer,
        )

        # 挂载指标收集器（通过 setattr 而非构造函数参数注入，
        # 避免 Agent 构造函数参数膨胀，保持核心接口简洁）
        if self._metrics:
            agent._metrics_collector = self._metrics

        # 注册配置热加载回调：重载时重建 LLM 客户端
        if self.llm is not None:

            def on_config_reload(new_config):
                from kocor.llm_provider.llm_factory import LlmFactory
                agent.llm = LlmFactory.create()

            Config.register_reload_hook(on_config_reload)

        # 注册进程退出清理
        atexit.register(self.tool_manager.stop_cron_scheduler)
        return agent