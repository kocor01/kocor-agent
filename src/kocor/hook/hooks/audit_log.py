"""内置钩子：工具调用审计日志。"""

import json
import os
from datetime import datetime

from kocor.hook.base import HookPoint, HookContext, HookResult


class AuditLogHook:
    """将所有工具调用记录到 JSONL 审计文件。"""

    hook_point = HookPoint.POST_TOOL

    def __init__(self, log_path: str = "./log/audit.log"):
        self.log_path = log_path

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

        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return HookResult(action="continue")
