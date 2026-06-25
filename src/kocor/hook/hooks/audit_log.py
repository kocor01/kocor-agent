"""内置钩子：工具调用审计日志。"""

import json
from datetime import datetime

from kocor.harness.logger import get_logger
from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction


class AuditLogHook:
    """将所有工具调用记录到 Logger 审计日志。"""

    hook_point = HookPoint.POST_TOOL

    def run(self, context: HookContext) -> HookResult:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": context.iteration,
        }
        if context.tool_call:
            entry["tool"] = context.tool_call.function.name
            entry["arguments"] = context.tool_call.function.arguments
            entry["tool_call_id"] = context.tool_call.id
        if context.error:
            entry["error"] = str(context.error)

        get_logger().info(json.dumps(entry, ensure_ascii=False))

        return HookResult(action=HookAction.CONTINUE)
