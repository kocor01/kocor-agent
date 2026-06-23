"""钩子系统核心类型 — HookPoint、HookContext、HookResult、Hook 协议。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class HookPoint(Enum):
    """可以注册钩子的生命周期节点。"""

    PRE_GENERATE = "pre_generate"      # LLM 生成前
    POST_GENERATE = "post_generate"    # LLM 生成后
    PRE_TOOL = "pre_tool"              # 工具执行前
    POST_TOOL = "post_tool"            # 工具执行后
    ON_ERROR = "on_error"              # 发生错误时
    ON_BUDGET_EXHAUSTED = "on_budget_exhausted"  # 预算耗尽时


@dataclass
class HookContext:
    """传递给钩子的上下文，包含执行时的环境信息。"""

    iteration: int
    messages: list
    tool_call: any = None
    tool_result: any = None
    error: Exception | None = None
    config: dict = field(default_factory=dict)


@dataclass
class HookResult:
    """钩子执行后返回的结果。"""

    action: str = "continue"  # continue | skip_tool | abort
    message: str = ""


class Hook(Protocol):
    """钩子实现的协议。"""

    @property
    def hook_point(self) -> HookPoint:
        ...

    def run(self, context: HookContext) -> HookResult:
        ...