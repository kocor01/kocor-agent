"""工具定义。

用于 JSON Schema 描述的工具定义。
"""

from __future__ import annotations


class ToolDefinition:
    """工具定义，用于 JSON Schema 描述。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 参数定义
        safety_level: 安全等级
    """

    def __init__(self, name: str, description: str, parameters: dict, safety_level: str = "caution"):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.safety_level = safety_level