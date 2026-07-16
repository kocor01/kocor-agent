"""HookManager — 钩子注册与执行管理器。"""

from __future__ import annotations

from kocor.hook.base import Hook, HookAction, HookContext, HookPoint, HookResult
from kocor.logger import Logger


class HookManager:
    """管理跨生命周期节点的钩子注册和执行。"""

    def __init__(self):
        self._hooks: dict[HookPoint, list] = {}

    def register_all(self, logger: Logger) -> None:
        """注册默认钩子。

        Args:
            logger: Logger 实例，注入到各钩子中。
        """
        from kocor.hook.hooks.audit_log import AuditLogHook
        self.register(AuditLogHook(logger=logger))

    def register(self, hook: Hook) -> None:
        """注册一个钩子实例。"""
        point = hook.hook_point
        if point not in self._hooks:
            self._hooks[point] = []
        self._hooks[point].append(hook)

    def unregister(self, hook: Hook) -> None:
        """移除特定的钩子实例。"""
        point = hook.hook_point
        if point in self._hooks:
            self._hooks[point] = [h for h in self._hooks[point] if h is not hook]

    def run(self, point: HookPoint, context: HookContext) -> list[HookResult]:
        """执行指定生命周期点的所有钩子。

        如果有钩子返回 action='abort'，则跳过剩余钩子。
        钩子异常会被捕获并记录为 continue 结果。
        """
        results: list[HookResult] = []
        for hook in self._hooks.get(point, []):
            try:
                result = hook.run(context)
                results.append(result)
                if result.action == HookAction.ABORT:
                    break
            except Exception as e:
                results.append(HookResult(
                    action=HookAction.CONTINUE,
                    message=f"Hook error: {e}",
                ))
        return results

    def clear(self) -> None:
        """移除所有注册的钩子。"""
        self._hooks.clear()
