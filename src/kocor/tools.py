"""工具系统。

工具注册与执行中心，提供内置工具（读文件、写文件、沙盒执行 Python）。
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable

from kocor.config import LLMConfig
from kocor.llm_client import ToolDefinition
from kocor.message import ToolCall, ToolResult


class ToolRegistry:
    """工具注册与执行中心。

    Attributes:
        _tools: 工具定义映射
        _handlers: 工具处理器映射
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}

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
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {type(e).__name__}: {e}",
            )


def create_default_tools(config: LLMConfig | None = None) -> ToolRegistry:
    """创建默认工具集（读文件、写文件、沙盒执行 Python）。

    Args:
        config: 可选配置

    Returns:
        已注册内置工具的 ToolRegistry
    """
    registry = ToolRegistry()
    timeout = config.timeout if config else 30

    registry.register(
        name="read_file",
        description="读取文件内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
        handler=_read_file,
    )

    registry.register(
        name="write_file",
        description="写入文件内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
    )

    registry.register(
        name="run_python",
        description="在沙盒中执行 Python 代码",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 代码"},
            },
            "required": ["code"],
        },
        handler=lambda **kwargs: _run_python(kwargs.get("code", ""), timeout),
    )

    return registry


def _read_file(path: str) -> str:
    """读取文件内容"""
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, content: str) -> str:
    """写入文件内容"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Success: wrote {len(content)} bytes to {path}"


def _run_python(code: str, timeout: int = 30) -> str:
    """在子进程中执行 Python 代码。

    Args:
        code: Python 代码字符串
        timeout: 超时秒数

    Returns:
        执行结果（stdout 或 stderr）
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output = f"Exit code: {result.returncode}\n{output}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
