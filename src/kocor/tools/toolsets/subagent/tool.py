"""Subagent 工具定义与 handler。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from kocor.tools.permission import PermissionManager

if TYPE_CHECKING:
    from kocor.tools.toolsets.subagent.runner import SubagentRunner


class SubagentTool:
    """子代理工具定义。

    工具注册时 handler 通过闭包注入 SubagentRunner 实例。
    """

    NAME = "subagent"
    DESCRIPTION = (
        "派生一个隔离上下文的子代理完成子任务，只返回摘要。"
        "适用于推理密集/会产生大量中间结果的子任务（深度搜索、多文件审查、分步调试），"
        "以保护主上下文。支持批量并行。"
    )
    SAFETY_LEVEL = PermissionManager.SAFETY_CAUTION
    PARAMETERS = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "单个子任务的目标描述（与 tasks 二选一）",
            },
            "context": {
                "type": "string",
                "description": "传给子代理的背景信息（父历史不会传入，必要信息须在此显式提供）",
            },
            "tasks": {
                "type": "array",
                "description": "批量并行子任务（与 goal 二选一，长度 ≤ max_concurrent，超限整批拒绝）",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["goal"],
                },
            },
        },
    }

    @staticmethod
    def handler(
        runner: SubagentRunner | None = None,
        goal: str | None = None,
        context: str | None = None,
        tasks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """执行子代理任务。

        Args:
            runner: SubagentRunner 实例（由 lambda 闭包注入）
            goal: 单任务目标
            context: 背景信息
            tasks: 批量任务列表

        Returns:
            JSON 字符串结果
        """
        if runner is None:
            return json.dumps({"status": "error", "summary": "SubagentRunner 未装配"}, ensure_ascii=False)

        result = runner.run(goal=goal, context=context, tasks=tasks)
        return json.dumps(result, ensure_ascii=False, default=str)