"""内置钩子：LLM 生成审计日志（token 消耗），写入独立 audit.log 文件。"""

import json
from datetime import datetime

from kocor.hook.base import HookAction, HookContext, HookPoint, HookResult
from kocor.logger import Logger


class AuditLogHook:
    """记录大模型每次生成的 token 消耗到 audit.log。

    通过主 Logger 的 ``audit()`` 方法写入，日志文件由 Logger 内部按类别分流。
    """

    hook_point = HookPoint.POST_GENERATE

    def __init__(self, logger: Logger):
        self._logger = logger

    def run(self, context: HookContext) -> HookResult:
        """记录本次 LLM 生成的 token 消耗到审计日志。

        从 context.response.usage 提取 token 计数，
        如无 usage 信息则标记为 "unavailable"。
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": context.iteration,
        }

        usage = None
        if context.response:
            usage = getattr(context.response, "usage", None)

        if usage:
            entry["prompt_tokens"] = usage.prompt_tokens
            entry["completion_tokens"] = usage.completion_tokens
            entry["total_tokens"] = usage.total_tokens
            entry["cached_tokens"] = usage.cached_tokens
        else:
            entry["usage"] = "unavailable"

        self._logger.audit(json.dumps(entry, ensure_ascii=False))

        return HookResult(action=HookAction.CONTINUE)