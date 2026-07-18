"""构建子代理的最小依赖 Loop。"""

from __future__ import annotations

from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import EventEmitter
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message
from kocor.loop import Loop
from kocor.tools.permission import PermissionManager
from kocor.tools.tool_manager import ToolManager
from kocor.tools.toolsets.subagent.system_prompt import build_subagent_system_prompt
from kocor.tools.toolsets.todo_tool import TodoStore


def _build_child_tool_manager(
    blocked_tools: tuple[str, ...],
    include_subagent: bool = False,
) -> ToolManager:
    """构建收窄的子代理 ToolManager。

    从完整内置工具集中剥离：cron（always）、blocked_tools（config 控制）、
    以及非 orchestrator 时剥离 subagent。

    Args:
        blocked_tools: 额外屏蔽的工具名称集合
        include_subagent: 是否注册 subagent 工具（orchestrator 保留）

    Returns:
        收窄后的 ToolManager 实例
    """
    tm = ToolManager()
    tm.register_builtin_tools(include_cron=False)
    tm.todo_store = TodoStore()  # 独立的 todo 列表，不与父代理共享

    # 剥离 blocked_tools
    for name in blocked_tools:
        if name in tm._handlers:
            del tm._tools[name]
            del tm._handlers[name]

    # orchestrator 角色注册 subagent 工具（handler 由 SubagentRunner 后续注入）
    if include_subagent:
        from kocor.tools.toolsets.subagent.tool import SubagentTool
        tm.register(
            name=SubagentTool.NAME,
            description=SubagentTool.DESCRIPTION,
            parameters=SubagentTool.PARAMETERS,
            handler=lambda **kw: "Error: subagent 工具尚未装配运行器",
            safety_level=SubagentTool.SAFETY_LEVEL,
            timeout=0,
        )

    return tm


def assemble_child_loop(
    goal: str,
    context: str | None,
    parent_llm: LLMClient,
    parent_tool_manager: ToolManager,
    depth: int,
    max_iterations: int | None = None,
    blocked_tools: tuple[str, ...] | None = None,
    auto_approve: bool | None = None,
    max_depth: int = 1,
) -> Loop:
    """组装一个完整的最小子代理 Loop，准备好运行。

    Args:
        goal: 子任务目标
        context: 背景上下文
        parent_llm: 父代理的 LLM 客户端（子代理继承复用）
        parent_tool_manager: 父代理的 ToolManager（用于派生工具集，但其本身不直接传给子代理）
        depth: 当前子代理深度（0=顶层子代理）
        max_iterations: 子代理迭代预算，None 则使用 Config 默认
        blocked_tools: 额外屏蔽工具，None 则使用 Config 默认
        auto_approve: 危险命令自动审批，None 则使用 Config 默认
        max_depth: 配置的最大嵌套深度，用于决定子代理角色

    Returns:
        已 seed 消息、可立即通过 run_messages() 驱动的 Loop
    """
    cfg = Config.load()
    max_iterations = max_iterations or cfg.subagent_max_iterations
    blocked_tools = blocked_tools or cfg.subagent_blocked_tools
    auto_approve = auto_approve if auto_approve is not None else cfg.subagent_auto_approve

    # 1. 角色判定
    is_orchestrator = (depth + 1) < max_depth

    # 2. 收窄 ToolManager（独立实例，不污染父进程）
    child_tm = _build_child_tool_manager(
        blocked_tools=blocked_tools,
        include_subagent=is_orchestrator,
    )

    # 3. 非交互权限
    child_tm.permission_mgr.policy = PermissionManager.POLICY_NONINTERACTIVE
    # 非交互策略从 Config 读取 subagent_auto_approve，
    # 但为了覆盖当前子代理的 auto_approve 值，临时写入 Config。
    # 注意：这是线程安全的写入（Config 是全局单例），多个子代理并发时会竞争。
    # 但 subagent_auto_approve 在子代理生命周期内不变，写入相同值无副作用。
    cfg.subagent_auto_approve = auto_approve

    # 4. 空 HookManager（子代理不重复审计、不打断父钩子链）
    hook_mgr = HookManager()

    # 5. 子代理 EventEmitter（空，无订阅者——子代理内部事件静默不上抛）
    child_emitter = EventEmitter()

    # 6. 构建聚焦系统提示
    workspace = None
    if parent_tool_manager._env is not None:
        workspace = parent_tool_manager._env.cwd
    system_prompt = build_subagent_system_prompt(
        goal=goal,
        context=context,
        workspace=workspace,
        is_orchestrator=is_orchestrator,
        depth=depth,
    )

    # 7. 构建 ContextManager 并手动 seed 消息
    child_ctx = ContextManager(
        tools=child_tm,
        memory=None,  # 子代理不加载父级记忆
        todo_store=child_tm.todo_store,
    )
    user_message = goal
    if context and context.strip():
        user_message = f"{goal}\n\n上下文:\n{context.strip()}"
    child_ctx.messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]
    child_ctx.tool_definitions = child_tm.get_definitions()

    # 8. 构建 Loop
    child_loop = Loop(
        llm=parent_llm,
        context=child_ctx,
        tool_manager=child_tm,
        hook_manager=hook_mgr,
        event_emitter=child_emitter,
        max_iterations=max_iterations,
    )

    return child_loop