"""钩子系统核心类型 — HookPoint、HookContext、HookResult、Hook 协议。"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class HookAction(str, Enum):
    """钩子执行结果动作常量。"""

    CONTINUE = "continue"          # 继续执行
    ABORT = "abort"                # 终止当前阶段


class HookPoint(Enum):
    """可以注册钩子的生命周期节点。"""

    PRE_GENERATE = "pre_generate"      # LLM 生成前
    POST_GENERATE = "post_generate"    # LLM 生成后
    PRE_TOOL = "pre_tool"              # 工具执行前
    POST_TOOL = "post_tool"            # 工具执行后
    PRE_SUMMARIZE = "pre_summarize"    # 摘要生成前
    POST_SUMMARIZE = "post_summarize"  # 摘要生成后
    ON_ERROR = "on_error"              # 发生错误时
    ON_BUDGET_EXHAUSTED = "on_budget_exhausted"  # 预算耗尽时


# 已知的 HookContext 字段名，用于 __init__ 中拆分未知字段
_KNOWN_HOOK_CONTEXT_FIELDS = frozenset({
    "iteration", "messages", "tool_call", "tool_result",
    "response", "error", "config", "extra",
})


class HookContext:
    """传递给钩子的上下文，包含执行时的环境信息。

    未知关键字参数自动归入 extra 字典，不会抛 TypeError。
    """

    def __init__(self, **kwargs):
        self.iteration: int = kwargs.get("iteration", 0)
        self.messages: list = kwargs.get("messages", [])
        self.tool_call: Any = kwargs.get("tool_call")
        self.tool_result: Any = kwargs.get("tool_result")
        self.response: Any = kwargs.get("response")
        self.error: Exception | None = kwargs.get("error")
        self.config: dict = kwargs.get("config", {})
        # 未知字段自动归入 extra
        self.extra: dict = {}
        for k, v in kwargs.items():
            if k not in _KNOWN_HOOK_CONTEXT_FIELDS:
                self.extra[k] = v
        # merge 显式传入的 extra
        if "extra" in kwargs and isinstance(kwargs["extra"], dict):
            self.extra.update(kwargs["extra"])


@dataclass
class HookResult:
    """钩子执行后返回的结果。"""

    action: HookAction = HookAction.CONTINUE
    message: str = ""


class Hook(Protocol):
    """钩子实现的协议。"""

    @property
    def hook_point(self) -> HookPoint:
        """返回钩子注册的触发点。"""
        ...

    def run(self, context: HookContext) -> HookResult:
        """执行钩子逻辑，返回执行结果。"""
        ...