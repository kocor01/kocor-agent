"""工具注册协议和基础设施。

提供 BundledTool 基类，统一内置工具的注册元数据和依赖注入方式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar


class BundledTool(ABC):
    """自注册工具基类。

    每个内置工具继承此类，提供注册元数据和 handler_factory 方法。
    ToolManager 通过 register_all_bundled() 一次性注册所有 BundledTool 子类。

    子类只需定义：
    - NAME / DESCRIPTION / PARAMETERS / SAFETY_LEVEL 类变量
    - handler_factory(cls, **deps) → Callable 类方法
    """

    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    PARAMETERS: ClassVar[dict] = {}
    SAFETY_LEVEL: ClassVar[str] = "caution"
    TIMEOUT: ClassVar[int | None] = None  # None = 继承 Config.tool_timeout

    @classmethod
    @abstractmethod
    def handler_factory(cls, **deps: Any) -> Callable:
        """创建带注入依赖的 handler 函数。

        Args:
            **deps: ToolManager 注入的共享依赖
                - file_state: FileStateTracker
                - env: LocalEnvironment
                - memory_store: MemoryStore | None
                - todo_store: TodoStore | None
                - subagent_runner: SubagentRunner | None

        Returns:
            可调用的 handler 函数，接收 **kwargs 返回 str
        """