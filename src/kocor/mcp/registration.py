"""MCP 工具注册与集成。"""

from __future__ import annotations

import json
import os
from typing import Any

from kocor.mcp.client import MCPClient
from kocor.mcp.config import load_mcp_servers, sanitize_server_name
from kocor.mcp.event_loop import MCPError


def _load_global_config(config_path: str) -> tuple[dict, dict]:
    """从 MCP 配置文件中加载全局设置。"""
    if not config_path or not os.path.exists(config_path):
        return {}, {}

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, {}

    return data.get("tool_output", {}), data.get("permissions", {})


def _build_handler(client, tool_name, prefix, server_name,
                   permission_manager, truncate_cfg):
    """构建带权限检查和输出截断的 handler。"""
    from kocor.mcp.truncate import truncate_output

    full_name = f"mcp_{prefix}_{sanitize_server_name(tool_name)}"

    def handler(**kwargs):
        if not permission_manager.check(full_name, server_name):
            return "[Permission Denied] 用户拒绝了此工具调用"
        raw = client.call_tool(tool_name, kwargs)
        return truncate_output(raw, truncate_cfg)

    return handler


def register_mcp_tools(toolRegistry, config_path: str = "") -> list:
    """将 MCP 服务器工具注册到指定 ToolRegistry。

    Args:
        toolRegistry: 目标 ToolRegistry 实例
        config_path: MCP 配置文件路径

    Returns:
        connected_clients: 用于关闭的客户端列表
    """
    servers = load_mcp_servers(config_path)

    # 全局配置
    tool_output_cfg, permissions_cfg = _load_global_config(config_path)

    from kocor.mcp.truncate import TruncateConfig
    truncate_cfg = TruncateConfig(
        max_bytes=tool_output_cfg.get("max_bytes", 50_000),
        max_lines=tool_output_cfg.get("max_lines", 2_000),
        max_line_length=tool_output_cfg.get("max_line_length", 2_000),
    )

    from kocor.mcp.permission import PermissionManager, PermissionPolicy
    server_policies = {}
    for name, cfg in servers.items():
        perm = permissions_cfg.get(name, {})
        server_policies[name] = PermissionPolicy(
            policy=perm.get("policy", "always_allow"),
            allowed_tools=perm.get("allowed_tools", []),
        )
    permission_manager = PermissionManager(server_policies)

    clients: list = []

    for name, cfg in servers.items():
        try:
            client = MCPClient(name, cfg)
            client.start()
            tool_list = client.list_tools()

            prefix = sanitize_server_name(name)
            for t in tool_list:
                handler = _build_handler(
                    client, t["name"], prefix, name,
                    permission_manager, truncate_cfg,
                )
                toolRegistry.register(
                    name=f"mcp_{prefix}_{sanitize_server_name(t['name'])}",
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {"type": "object"}),
                    handler=handler,
                )

            clients.append(client)
        except MCPError:
            try:
                client.shutdown()
            except Exception:
                pass

    return clients


def shutdown_mcp_clients(clients: list) -> None:
    """逐个关闭 MCP 客户端。"""
    for client in clients:
        try:
            client.shutdown()
        except Exception:
            pass
