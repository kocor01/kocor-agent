"""MCP 测试基础设施：FakeMCPClient 用于集成测试。

不依赖真实的 MCP 服务器，通过模拟客户端验证 McpManager 的工具注册逻辑。
"""

from __future__ import annotations

import json
from typing import Any

from kocor.mcp.config import MCPConfig


class FakeMCPClient:
    """测试用假 MCP 客户端，不连接真实服务器。

    模拟 MCPClient 的公开接口（start/list_tools/call_tool/shutdown），
    用于 McpManager 的单元测试和集成测试。

    用法:
        mgr = McpManager(tool_manager, "")
        mgr.register_all(client_factory=FakeMCPClient)
    """

    def __init__(
        self,
        name: str,
        config: MCPConfig | dict | None = None,
        tools: list[dict] | None = None,
    ):
        self.name = name
        self.config = config if isinstance(config, dict) else {}
        self._tools = tools or [
            {
                "name": "read_file",
                "description": "Read a file from the filesystem",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write a file to the filesystem",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"},
                    },
                    "required": ["path", "content"],
                },
            },
        ]
        self._started = False
        self._shutdown_called = False

    def start(self) -> None:
        """模拟连接服务器。"""
        self._started = True

    def list_tools(self) -> list[dict]:
        """返回模拟工具列表。"""
        if not self._started:
            raise RuntimeError(f"MCP client '{self.name}' not started")
        return self._tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """模拟工具调用，返回 JSON 结果。"""
        if not self._started:
            raise RuntimeError(f"MCP client '{self.name}' not started")
        return json.dumps({"result": f"fake {name} called with {arguments}"})

    def shutdown(self) -> None:
        """模拟关闭连接。"""
        self._shutdown_called = True
        self._started = False


class FakeFailingMCPClient(FakeMCPClient):
    """模拟 MCP 连接失败的假客户端。"""

    def start(self) -> None:
        """模拟连接失败。"""
        self._started = False
        raise ConnectionError(f"Failed to connect to MCP server '{self.name}'")

    def list_tools(self) -> list[dict]:
        """未连接时不应调用此方法。"""
        raise RuntimeError(f"Client '{self.name}' not connected")

    def call_tool(self, name: str, arguments: dict) -> str:
        """未连接时不应调用此方法。"""
        raise RuntimeError(f"Client '{self.name}' not connected")

    def shutdown(self) -> None:
        """失败时关闭应安全。"""
        self._shutdown_called = True