"""MCP (Model Context Protocol) 客户端。"""

from __future__ import annotations

import os
from typing import Any

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from kocor.mcp.config import MCPConfig
from kocor.mcp.event_loop import MCPError, _run_async


class MCPClient:
    """MCP 客户端，通过官方 SDK 与 MCP 服务器通信。

    所有异步 SDK 操作通过后台事件循环桥接为同步调用。
    支持 stdio、Streamable HTTP、SSE 三种传输。
    """

    def __init__(self, name: str, config: MCPConfig):
        self._name = name
        self._config = config
        self._session: ClientSession | None = None
        self._transport_exit_stack: list[Any] = []

    # ── 同步公开 API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """连接服务器并完成 MCP 握手。"""
        _run_async(self._async_start(), timeout=self._config.connect_timeout or 30)

    def list_tools(self) -> list[dict]:
        """获取工具列表。

        Returns:
            [{"name", "description", "inputSchema", ...}, ...]
        """
        result = _run_async(self._async_list_tools(), timeout=self._config.timeout)
        tools = result.tools
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {"type": "object"},
            }
            for t in tools
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        """调用工具，连接断开时自动重连一次。

        Args:
            name: 工具名
            arguments: 参数字典

        Returns:
            文本形式的结果
        """
        try:
            result = self._do_call_tool(name, arguments)
        except MCPError:
            try:
                self.reconnect()
            except Exception:
                raise  # 重连失败，透传原错误
            result = self._do_call_tool(name, arguments)

        texts = [
            block.text
            for block in result.content
            if isinstance(block, mcp_types.TextContent)
        ]
        text = "\n".join(texts)

        if result.isError:
            return f"[MCP Error] {text}"
        return text

    def shutdown(self) -> None:
        """关闭连接，清理资源。"""
        try:
            _run_async(self._async_shutdown(), timeout=5)
        except Exception:
            pass

    def reconnect(self) -> None:
        """断开当前连接并重新建立。"""
        _run_async(self._async_shutdown(), timeout=5)
        _run_async(self._async_start(), timeout=self._config.connect_timeout or 30)

    def _do_call_tool(self, name: str, arguments: dict) -> mcp_types.CallToolResult:
        """执行工具调用的内部方法，不做重试。"""
        return _run_async(
            self._async_call_tool(name, arguments),
            timeout=self._config.timeout,
        )

    # ── 异步内部实现 ──────────────────────────────────────────────────────

    async def _async_start(self) -> None:
        """在事件循环上创建传输层和会话。"""
        read_stream = None
        write_stream = None

        if self._config.url:
            url = self._config.url
            headers = dict(self._config.headers or {})

            if self._config.transport == "sse":
                cm = sse_client(url, headers=headers or None,
                                sse_read_timeout=300.0)
            else:
                cm = streamable_http_client(url)

            read_stream, write_stream = await cm.__aenter__()
            self._transport_exit_stack.append(cm)
        else:
            cmd = self._config.command
            if not cmd:
                raise MCPError(f"Server '{self._name}': empty command")

            params = StdioServerParameters(
                command=cmd,
                args=self._config.args or [],
                env=self._build_safe_env(),
            )
            cm = stdio_client(params)
            read_stream, write_stream = await cm.__aenter__()
            self._transport_exit_stack.append(cm)

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        self._transport_exit_stack.append(self._session)

        init_result = await self._session.initialize()
        proto = init_result.protocolVersion or ""
        if proto < "2024-11-05":
            raise MCPError(
                f"Server '{self._name}' protocol version '{proto}' is too old"
            )

    async def _async_list_tools(self) -> mcp_types.ListToolsResult:
        """列表工具的异步实现。"""
        if self._session is None:
            raise MCPError(f"Server '{self._name}' is not connected")
        return await self._session.list_tools()

    async def _async_call_tool(self, name: str, arguments: dict) -> mcp_types.CallToolResult:
        """调用工具的异步实现。"""
        if self._session is None:
            raise MCPError(f"Server '{self._name}' is not connected")
        return await self._session.call_tool(name, arguments=arguments)

    async def _async_shutdown(self) -> None:
        """关闭连接的异步实现。"""
        for item in reversed(self._transport_exit_stack):
            try:
                await item.__aexit__(None, None, None)
            except Exception:
                pass
        self._transport_exit_stack.clear()
        self._session = None

    def _build_safe_env(self) -> dict[str, str]:
        """构建安全的环境变量（过滤敏感 Key）。"""
        env = os.environ.copy()
        sensitive_keys = [
            key for key in env
            if key.endswith(("_API_KEY", "_SECRET", "_TOKEN"))
            or key in ("OPENAI_ORG_ID",)
        ]
        for key in sensitive_keys:
            env.pop(key, None)
        env.update(self._config.env)
        return env
