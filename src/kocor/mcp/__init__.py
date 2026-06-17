"""MCP (Model Context Protocol) 模块。"""

from kocor.mcp.client import MCPClient
from kocor.mcp.config import MCPConfig, load_mcp_servers, sanitize_server_name
from kocor.mcp.event_loop import MCPError
from kocor.mcp.permission import PermissionManager, PermissionPolicy
from kocor.mcp.registration import register_mcp_tools, shutdown_mcp_clients
from kocor.mcp.truncate import TruncateConfig, truncate_output

__all__ = [
    "MCPClient",
    "MCPConfig",
    "MCPError",
    "PermissionManager",
    "PermissionPolicy",
    "TruncateConfig",
    "load_mcp_servers",
    "register_mcp_tools",
    "sanitize_server_name",
    "shutdown_mcp_clients",
    "truncate_output",
]
