"""Skill 核心数据模型。

定义技能类型枚举、触发策略枚举、技能定义数据类等核心类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from kocor.tools.permission import PermissionManager

from kocor.tools.tool_manager import ToolManager


class SkillType(Enum):
    """技能类型。"""

    PROMPT = "prompt"  # Prompt 模板，LLM 编排工具调用
    CODE = "code"  # Python 函数，直接执行


class InvokeStrategy(Enum):
    """技能触发策略。"""

    SLASH = "slash"  # 仅 /name 触发
    LLM = "llm"  # 仅作为 tool 暴露给 LLM
    BOTH = "both"  # 两种方式均可


@dataclass
class SkillDefinition:
    """技能定义，对应一条配置或一个发现文件。

    Attributes:
        name: 技能唯一名称，用作 slash 命令名和 tool 名
        description: 技能描述（LLM 可见）
        skill_type: 技能类型（PROMPT / CODE）
        invoke_strategy: 触发策略（SLASH / LLM / BOTH）
        prompt_template: PROMPT 技能的模板字符串
        prompt_role: PROMPT 技能注入的角色（system / user）
        handler: CODE 技能的可调用对象
        parameters: CODE 技能的 JSON Schema 参数定义
        category: 分类标签
        enabled: 是否启用
        version: 版本号
        author: 作者
    """

    name: str
    description: str
    skill_type: SkillType

    invoke_strategy: InvokeStrategy = InvokeStrategy.BOTH

    prompt_template: str = ""
    prompt_role: str = "user"

    handler: Callable | None = None
    parameters: dict | None = None

    category: str = "general"
    enabled: bool = True
    version: str = "1.0.0"
    author: str = ""
    safety_level: str = PermissionManager.SAFETY_CAUTION


@dataclass
class SkillContext:
    """技能执行上下文，由调用方在调用时构造传入。"""

    user_input: str
    tool_manager: ToolManager | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class SkillResult:
    """技能执行结果。"""

    content: str
    skill_name: str
    success: bool = True
    error: str | None = None