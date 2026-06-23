"""Kocor Agent Harness — Agent 生命周期管理的运行时系统。"""

from kocor.harness.loop import ToolCallRecord
from kocor.harness.budget import IterationBudget
from kocor.harness.events import HarnessEvent, EventEmitter
from kocor.harness.config import HarnessConfig
from kocor.harness.permission import (
    PermissionManager, ToolSafetyLevel, TOOL_SAFETY_MAP,
)
from kocor.harness.file_guard import FileAccessGuard
from kocor.harness.sandbox import Sandbox, SandboxResult
from kocor.harness.error_handler import ErrorHandler, GracefulDegradation
from kocor.harness.debug import DebugManager
from kocor.harness.logger import HarnessLogger

__all__ = [
    # 循环
    "ToolCallRecord",
    # 预算
    "IterationBudget",
    # 事件
    "HarnessEvent",
    "EventEmitter",
    # 配置
    "HarnessConfig",
    # 权限
    "PermissionManager",
    "ToolSafetyLevel",
    "TOOL_SAFETY_MAP",
    # 文件守卫
    "FileAccessGuard",
    # 沙箱
    "Sandbox",
    "SandboxResult",
    # 错误处理
    "ErrorHandler",
    "GracefulDegradation",
    # 可观测性
    "DebugManager",
    "HarnessLogger",
]