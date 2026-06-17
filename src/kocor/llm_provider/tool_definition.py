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
    """

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict:
        """转换为 OpenAI API 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }