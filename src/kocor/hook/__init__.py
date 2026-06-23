"""钩子系统 — 生命周期钩子核心类型和内置钩子。"""

from kocor.hook.base import HookPoint, HookContext, HookResult, Hook
from kocor.hook.hook_manager import HookManager
from kocor.hook.hooks import AuditLogHook

__all__ = [
    "HookPoint",
    "HookContext",
    "HookResult",
    "Hook",
    "HookManager",
    "AuditLogHook",
]