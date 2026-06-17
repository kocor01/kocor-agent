"""MCP (Model Context Protocol) 客户端。

基于官方 MCP Python SDK，通过后台事件循环桥接异步 SDK 到同步接口。
零侵入集成：通过 ToolRegistry.register() API 注册 MCP 工具，Agent 核心零感知。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from typing import Any

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

# ── 后台事件循环 ───────────────────────────────────────────────────────────

_mcp_loop: asyncio.AbstractEventLoop | None = None
_mcp_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """获取或创建后台 asyncio 事件循环（守护线程）。"""
    global _mcp_loop
    if _mcp_loop is None:
        with _mcp_loop_lock:
            if _mcp_loop is None:
                _mcp_loop = asyncio.new_event_loop()
                t = threading.Thread(
                    target=_mcp_loop.run_forever,
                    daemon=True,
                    name="mcp-event-loop",
                )
                t.start()
    return _mcp_loop


def _run_async(coro, timeout: float = 120) -> Any:
    """在后台事件循环上运行协程，同步等待结果。

    Args:
        coro: 要执行的协程
        timeout: 超时秒数

    Returns:
        协程返回值

    Raises:
        MCPError: 超时或执行出错
    """
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        raise MCPError(f"Async operation timed out after {timeout}s")
    except Exception as e:
        if isinstance(e, MCPError):
            raise
        raise MCPError(str(e)) from e


# ── 数据模型 ────────────────────────────────────────────────────────────────


class MCPError(Exception):
    """MCP 协议错误。"""
    pass


@dataclass
class MCPConfig:
    """单个 MCP 服务器的配置。

    stdio 服务器使用 command/args/env。
    远程服务器使用 url/headers/transport。
    """

    # stdio 字段
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # 远程字段
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    transport: str = "streamablehttp"  # "streamablehttp" | "sse"
    # 超时
    timeout: int = 120
    connect_timeout: int = 30


# ── 配置加载 ────────────────────────────────────────────────────────────────


def load_mcp_servers(config_path: str) -> dict[str, MCPConfig]:
    """从 JSON 配置文件加载 MCP 服务器列表。"""
    if not config_path or not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    servers = data.get("mcpServers", {})
    result = {}
    for name, cfg in servers.items():
        result[name] = MCPConfig(
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            env=cfg.get("env", {}),
            url=cfg.get("url", ""),
            headers=cfg.get("headers", {}),
            transport=cfg.get("transport", "streamablehttp"),
            timeout=cfg.get("timeout", 120),
            connect_timeout=cfg.get("connect_timeout", 30),
        )
    return result


def sanitize_server_name(name: str) -> str:
    """将服务器/工具名称清洗为安全的 Python 标识符。"""
    result = []
    for ch in name.lower():
        if ch.isalnum() or ch == "_":
            result.append(ch)
        else:
            result.append("_")
    return "".join(result)


# ── MCP 客户端 ──────────────────────────────────────────────────────────────


class MCPClient:
    """MCP 客户端，通过官方 SDK 与 MCP 服务器通信。

    所有异步 SDK 操作通过后台事件循环桥接为同步调用。
    支持 stdio、Streamable HTTP、SSE 三种传输。
    """

    def __init__(self, name: str, config: MCPConfig):
        self._name = name
        self._config = config
        self._session: ClientSession | None = None
        self._transport_exit_stack: list[Any] = []  # 用于 shutdown 时清理

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
        """调用工具。

        Args:
            name: 工具名
            arguments: 参数字典

        Returns:
            文本形式的结果
        """
        result = _run_async(
            self._async_call_tool(name, arguments),
            timeout=self._config.timeout,
        )

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

    # ── 异步内部实现 ──────────────────────────────────────────────────────

    async def _async_start(self) -> None:
        """在事件循环上创建传输层和会话。"""
        read_stream = None
        write_stream = None

        if self._config.url:
            # 远程服务器
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
            # stdio 服务器
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
        # 逆序关闭（session 先关，transport 后关）
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


# ── 工具注册与集成 ─────────────────────────────────────────────────────────


def _load_global_config(config_path: str) -> tuple[dict, dict]:
    """从 MCP 配置文件中加载全局设置。"""
    if not config_path or not os.path.exists(config_path):
        return {}, {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, {}

    return data.get("tool_output", {}), data.get("permissions", {})


def _build_handler(client, tool_name, prefix, server_name,
                   permission_manager, truncate_cfg):
    """构建带权限检查和输出截断的 handler。"""
    from kocor.mcp_truncate import TruncateConfig, truncate_output

    full_name = f"mcp_{prefix}_{sanitize_server_name(tool_name)}"
    if truncate_cfg is None:
        truncate_cfg = TruncateConfig()

    def handler(**kwargs):
        if not permission_manager.check(full_name, server_name):
            return "[Permission Denied] 用户拒绝了此工具调用"
        raw = client.call_tool(tool_name, kwargs)
        return truncate_output(raw, truncate_cfg)

    return handler


def register_mcp_tools(config_path: str = "") -> tuple[list, list]:
    """连接所有 MCP 服务器，创建独立的 MCP ToolRegistry 并注册工具。

    与 `create_default_tools()` 独立 — 调用者需手动合并：
        tools = create_default_tools(config)
        mcp_registry, mcp_clients = register_mcp_tools(config.mcp_config)
        tools.merge(mcp_registry)

    Args:
        config_path: MCP 配置文件路径

    Returns:
        (mcp_tool_registry, connected_clients)
        - mcp_tool_registry: 包含所有 MCP 工具的 ToolRegistry
        - connected_clients: 用于关闭的客户端列表
    """
    from kocor.tools import ToolRegistry

    servers = load_mcp_servers(config_path)

    # 全局配置
    tool_output_cfg, permissions_cfg = _load_global_config(config_path)

    from kocor.mcp_truncate import TruncateConfig
    truncate_cfg = TruncateConfig(
        max_bytes=tool_output_cfg.get("max_bytes", 50_000),
        max_lines=tool_output_cfg.get("max_lines", 2_000),
        max_line_length=tool_output_cfg.get("max_line_length", 2_000),
    )

    from kocor.mcp_permission import PermissionManager, PermissionPolicy
    server_policies = {}
    for name, cfg in servers.items():
        perm = permissions_cfg.get(name, {})
        server_policies[name] = PermissionPolicy(
            policy=perm.get("policy", "always_allow"),
            allowed_tools=perm.get("allowed_tools", []),
        )
    permission_manager = PermissionManager(server_policies)

    mcp_registry = ToolRegistry()
    clients: list = []

    for name, cfg in servers.items():
        client = MCPClient(name, cfg)
        try:
            client.start()
            tool_list = client.list_tools()

            prefix = sanitize_server_name(name)
            for t in tool_list:
                tname = sanitize_server_name(t.get("name", "unknown"))
                handler = _build_handler(
                    client, t["name"], prefix, name,
                    permission_manager, truncate_cfg,
                )
                mcp_registry.register(
                    name=f"mcp_{prefix}_{tname}",
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {"type": "object"}),
                    handler=handler,
                )

            clients.append(client)
        except Exception:
            try:
                client.shutdown()
            except Exception:
                pass

    return mcp_registry, clients


def shutdown_mcp_clients(clients: list) -> None:
    """逐个关闭 MCP 客户端。"""
    for client in clients:
        try:
            client.shutdown()
        except Exception:
            pass
