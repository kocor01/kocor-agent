"""MCP 工具注册与集成管理。"""

from __future__ import annotations

import json
import logging
import os

from kocor.mcp.client import MCPClient
from kocor.mcp.config import load_mcp_servers, sanitize_server_name
from kocor.mcp.event_loop import MCPError
from kocor.tools.permission import PermissionManager

logger = logging.getLogger(__name__)


class McpManager:
    """管理 MCP 服务器的连接、工具注册和关闭。"""

    def __init__(self, tool_manager, config_path: str = ""):
        self.tool_manager = tool_manager
        self.config_path = config_path
        self._clients: list = []

    def register_all(
        self,
        client_factory=None,
    ) -> list:
        """连接所有 MCP 服务器并注册工具到 ToolManager。

        Args:
            client_factory: 客户端工厂类（测试时传入 FakeMCPClient）。
                为 None 时使用默认的 MCPClient。

        Returns:
            connected_clients: 用于关闭的客户端列表
        """
        servers = load_mcp_servers(self.config_path)
        if not servers:
            return self._clients

        permissions_cfg = self._load_permissions()
        factory = client_factory or MCPClient

        for name, cfg in servers.items():
            try:
                client = factory(name, cfg)
                client.start()
                tool_list = client.list_tools()

                prefix = sanitize_server_name(name)
                # 按工具安全等级反向查找
                server_perms = permissions_cfg.get(name, {})
                tool_safety: dict[str, str] = {}
                for level, tools in server_perms.items():
                    for tool_full_name in tools:
                        tool_safety[tool_full_name] = level

                for t in tool_list:
                    full_name = f"mcp_{prefix}_{sanitize_server_name(t['name'])}"

                    def handler(_tool_name=t["name"], _client=client, **kwargs):
                        try:
                            return _client.call_tool(_tool_name, kwargs)
                        except MCPError as e:
                            raise MCPError(
                                f"MCP server '{name}' tool '{_tool_name}' failed: {e}"
                            )

                    self.tool_manager.register(
                        name=full_name,
                        description=t.get("description", ""),
                        parameters=t.get("inputSchema", {"type": "object"}),
                        safety_level=tool_safety.get(full_name, PermissionManager.SAFETY_CAUTION),
                        handler=handler,
                    )

                self._clients.append(client)

            except (ConnectionError, MCPError, RuntimeError, Exception) as e:
                # 单个服务器失败不影响其他服务器注册
                logger.warning("MCP server '%s' registration failed: %s", name, e)
                try:
                    client.shutdown()
                except Exception:
                    pass

        return self._clients

    def shutdown_all(self) -> None:
        """逐个关闭所有 MCP 客户端。"""
        for client in self._clients:
            try:
                client.shutdown()
            except Exception:
                pass

    def _load_permissions(self) -> dict:
        if not self.config_path or not os.path.exists(self.config_path):
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        return data.get("permissions", {})

    