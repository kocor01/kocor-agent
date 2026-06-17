"""工具注册与执行中心。"""

from __future__ import annotations

import json
import os
from typing import Callable

from kocor.llm_provider.tool_definition import ToolDefinition
from kocor.message import ToolCall, ToolResult


class ToolRegistry:
    """工具注册与执行中心。

    Attributes:
        _tools: 工具定义映射
        _handlers: 工具处理器映射
        _timeout: 工具执行超时（秒）
    """

    def __init__(self, allowed_dir: str = "", timeout: int = 30):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._timeout = timeout

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[..., str],
    ) -> None:
        """注册工具。

        Args:
            name: 工具名称
            description: 工具描述
            parameters: JSON Schema 参数定义
            handler: 工具处理器，接收 **kwargs 返回结果字符串
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )
        self._handlers[name] = handler

    def get_definitions(self) -> list[ToolDefinition]:
        """返回所有工具的 ToolDefinition 列表"""
        return list(self._tools.values())

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用。

        Args:
            tool_call: 工具调用请求

        Returns:
            ToolResult: 执行结果
        """
        name = tool_call.function.name
        if name not in self._handlers:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: tool '{name}' not found",
            )

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: invalid JSON arguments for '{name}'",
            )

        try:
            result = self._handlers[name](**args)
            return ToolResult(tool_call_id=tool_call.id, content=str(result))
        except PermissionError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {e}",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {type(e).__name__}: {e}",
            )

    def merge(self, other: ToolRegistry) -> None:
        """合并另一个 ToolRegistry 的全部工具到当前实例。

        Args:
            other: 另一个 ToolRegistry（其工具和处理器将被导入）
        """
        for name in other._tools:
            self._tools[name] = other._tools[name]
            self._handlers[name] = other._handlers[name]
