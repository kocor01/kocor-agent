"""构建子代理聚焦系统提示。"""

from kocor.context.system_prompt import SystemPromptBuilder


def build_subagent_system_prompt(
    goal: str,
    context: str | None = None,
    workspace: str | None = None,
    is_orchestrator: bool = False,
    depth: int = 0,
) -> str:
    """构建子代理的聚焦系统提示。

    子代理不加载父级记忆/会话历史/技能，只以目标 + 上下文 + 工作目录
    构成最小化提示。orchestrator 角色会追加嵌套指导。

    Args:
        goal: 子任务目标
        context: 显式传递的背景信息（父历史不会自动传入）
        workspace: 绝对工作目录路径，避免子代理臆造容器路径
        is_orchestrator: 是否为 orchestrator 角色（可再委派）
        depth: 当前子代理深度（0=顶层，1=第一层子代理）

    Returns:
        聚焦系统提示字符串
    """
    lines = [
        "你是一个聚焦的子代理（subagent），在隔离上下文中执行一个明确的子任务。",
        "",
        "【你的任务】",
        goal.strip(),
    ]

    if context and context.strip():
        lines.extend(["", "【上下文】", context.strip()])

    if workspace:
        lines.extend(["", "【工作目录】", workspace])

    lines.extend([
        "",
        "【输出要求】",
        "完成后用一段紧凑摘要回复，包含：做了什么 / 关键发现 / 改动的文件 / 遗留问题。",
        "摘要要精炼——过长的摘要会挤占父代理上下文。不要复述中间工具结果。",
    ])

    # 注入项目指令（L2 层，来自 KOCOR.md / CLAUDE.md）
    try:
        project_instructions = SystemPromptBuilder._load_project_instructions()
        if project_instructions:
            lines.extend(["", project_instructions])
    except Exception:
        pass  # 项目指令加载失败不阻断子代理

    if is_orchestrator:
        lines.extend([
            "",
            "【子代理委派指导】",
            "你可以再委派子代理来并行或分担子任务，但注意：",
            f"  - 你的深度为 {depth}，委派出的子代理深度为 {depth + 1}。",
            "  - 委派适合：独立子任务、需要大量工具调用的任务。",
            "  - 不要委派过于简单的任务（直接做更快）。",
            "  - 确保 context 字段包含子代理需要的所有信息。",
        ])

    return "\n".join(lines)