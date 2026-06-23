"""工具注册与执行中心。"""

from __future__ import annotations

import json
import os
from typing import Callable

from kocor.tools.definitions import ToolDefinition
from kocor.llm_provider.message import ToolCall, ToolResult


class ToolManager:
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

    def register_defaults(self) -> None:
        """向当前 ToolManager 注册内置工具（读文件、写文件、沙盒执行 Python）。"""
        from kocor.tools.toolset import read_file, write_file, run_python

        read_file.toolRegistry_to(self)
        write_file.toolRegistry_to(self)
        run_python.toolRegistry_to(self)

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
