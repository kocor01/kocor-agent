"""工具系统。

工具注册与执行中心，提供内置工具（读文件、写文件、沙盒执行 Python）。
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable

from kocor.config import LLMConfig
from kocor.llm_client import ToolDefinition
from kocor.message import ToolCall, ToolResult


class ToolRegistry:
    """工具注册与执行中心。

    Attributes:
        _tools: 工具定义映射
        _handlers: 工具处理器映射
        _allowed_dir: 文件操作允许的根目录
        _timeout: 工具执行超时（秒）
    """

    def __init__(self, allowed_dir: str = "", timeout: int = 30):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._allowed_dir = allowed_dir or os.path.realpath(os.getcwd())
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


def create_default_tools(config: LLMConfig | None = None) -> ToolRegistry:
    """创建默认工具集（读文件、写文件、沙盒执行 Python）。

    Args:
        config: 可选配置

    Returns:
        已注册内置工具的 ToolRegistry
    """
    allowed_dir = os.path.realpath(os.getcwd())
    timeout = config.timeout if config else 30

    registry = ToolRegistry(allowed_dir=allowed_dir, timeout=timeout)

    # 创建闭包捕获 allowed_dir 和 timeout，避免全局可变状态
    def _read_file(path: str) -> str:
        safe_path = _resolve_safe_path(path, allowed_dir)
        if not os.path.exists(safe_path):
            return f"Error: file not found: {path}"
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(path: str, content: str) -> str:
        safe_path = _resolve_safe_path(path, allowed_dir)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: wrote {len(content)} bytes to {path}"

    def _run_python(code: str) -> str:
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_sanitize_env(),
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
        handler=_run_python,
    )

    return registry


def _resolve_safe_path(path: str, allowed_dir: str) -> str:
    """解析并校验路径是否在允许目录内，防止路径遍历攻击。

    Args:
        path: 用户传入的路径
        allowed_dir: 允许的根目录

    Returns:
        归一化后的安全绝对路径

    Raises:
        PermissionError: 路径尝试逃逸到允许目录外
    """
    resolved = os.path.realpath(os.path.join(allowed_dir, path))
    if resolved != allowed_dir and not resolved.startswith(allowed_dir + os.sep):
        raise PermissionError(f"Path traversal denied: {path}")
    return resolved


def _sanitize_env() -> dict[str, str]:
    """创建不含敏感凭证的环境变量副本，防止子进程泄露 API Key。"""
    env = os.environ.copy()
    _sensitive_keys = [key for key in env
                       if key.endswith(("_API_KEY", "_SECRET", "_TOKEN"))
                       or key in ("OPENAI_ORG_ID",)]
    for key in _sensitive_keys:
        env.pop(key, None)
    return env
