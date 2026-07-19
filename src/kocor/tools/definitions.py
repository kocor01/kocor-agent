"""工具定义。

用于 JSON Schema 描述的工具定义。
"""

from __future__ import annotations

from kocor.tools.permission import PermissionManager


class ToolDefinition:
    """工具定义，用于 JSON Schema 描述。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 参数定义
        safety_level: 安全等级
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        safety_level: str = PermissionManager.SAFETY_CAUTION,
        timeout: int | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.safety_level = safety_level
        # 工具级超时覆盖：None 由 ToolManager.register 在注册时解析为
        # Config.tool_timeout；0=不超时（供 subagent 等长生命周期工具）；
        # 正数=自定义秒数。
        self.timeout = timeout

    def __repr__(self):
        return f"ToolDefinition(name={self.name}, safety={self.safety_level})"