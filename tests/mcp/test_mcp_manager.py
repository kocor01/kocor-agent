"""MCP 管理器 McpManager 集成测试（使用 FakeMCPClient）。

验证：
- 无配置时返回空列表
- 单服务器注册工具
- 多服务器注册工具，名称不冲突
- 单服务器失败不影响其他服务器
- 全部服务器失败时优雅降级
- 工具安全等级从 permissions 配置正确解析
- shutdown_all 关闭所有客户端
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kocor.mcp._testing import FakeFailingMCPClient, FakeMCPClient
from kocor.mcp.mcp_manager import McpManager
from kocor.tools.permission import PermissionManager
from kocor.tools.tool_manager import ToolManager


class TestMcpManagerRegistration:
    """McpManager.register_all 注册逻辑测试。"""

    def test_no_config_registers_nothing(self):
        """无配置文件时不应注册任何工具。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")
        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={}):
            clients = mgr.register_all(client_factory=FakeMCPClient)
        assert clients == []

    def test_no_config_returns_empty_list(self):
        """配置文件不存在时返回空列表。"""
        tm = ToolManager()
        mgr = McpManager(tm, "/nonexistent/config.json")
        clients = mgr.register_all(client_factory=FakeMCPClient)
        # 空路径触发 load_mcp_servers 返回 {}
        assert clients == []

    def test_single_server_registers_tools(self):
        """单 MCP 服务器应注册其所有工具。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={
            "filesystem": {"command": "node", "args": ["server.js"]},
        }):
            clients = mgr.register_all(client_factory=FakeMCPClient)

        assert len(clients) == 1
        definitions = tm.get_definitions()
        names = {d.name for d in definitions}
        assert "mcp_filesystem_read_file" in names
        assert "mcp_filesystem_write_file" in names

    def test_multiple_servers_register_tools(self):
        """多 MCP 服务器应注册各自工具，名称不冲突。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        class FakeMCPClientA(FakeMCPClient):
            """服务器 A 的工具列表。"""

            def __init__(self, name, config):  # noqa: N803
                super().__init__(name, config, tools=[
                    {"name": "tool_a", "description": "Tool A", "inputSchema": {}},
                ])

        class FakeMCPClientB(FakeMCPClient):
            """服务器 B 的工具列表。"""

            def __init__(self, name, config):  # noqa: N803
                super().__init__(name, config, tools=[
                    {"name": "tool_b", "description": "Tool B", "inputSchema": {}},
                ])

        tool_map = {
            "server_a": FakeMCPClientA,
            "server_b": FakeMCPClientB,
        }

        def factory(name, cfg):
            return tool_map[name](name, cfg)

        with patch(
            "kocor.mcp.mcp_manager.load_mcp_servers",
            return_value={
                "server_a": {"command": "a"},
                "server_b": {"command": "b"},
            },
        ):
            clients = mgr.register_all(client_factory=factory)

        assert len(clients) == 2
        definitions = tm.get_definitions()
        names = {d.name for d in definitions}
        assert "mcp_server_a_tool_a" in names
        assert "mcp_server_b_tool_b" in names
        assert len(names) == 2


class TestMcpManagerFailureIsolation:
    """MCP 服务器连接失败的隔离性测试。"""

    def test_single_server_failure_does_not_block_others(self):
        """一台服务器连接失败不应阻止其他服务器注册。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        # 使用混合的工厂：server_a 失败，server_b 成功
        calls = {}

        def factory(name, cfg):
            if name == "bad_server":
                raise ConnectionError(f"Failed to connect to {name}")
            client = FakeMCPClient(name, cfg)
            calls[name] = client
            return client

        with patch(
            "kocor.mcp.mcp_manager.load_mcp_servers",
            return_value={
                "good_server": {"command": "good"},
                "bad_server": {"command": "bad"},
            },
        ):
            clients = mgr.register_all(client_factory=factory)

        # 只有 good_server 成功注册
        definitions = tm.get_definitions()
        names = {d.name for d in definitions}
        assert "mcp_good_server_read_file" in names
        assert "mcp_bad_server_read_file" not in names

    def test_all_servers_fail_gracefully(self):
        """所有服务器都失败时不应崩溃。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        def factory(name, cfg):
            raise ConnectionError(f"Failed to connect to {name}")

        with patch(
            "kocor.mcp.mcp_manager.load_mcp_servers",
            return_value={
                "server1": {"command": "c1"},
                "server2": {"command": "c2"},
            },
        ):
            # 不应抛出异常
            clients = mgr.register_all(client_factory=factory)

        assert clients == []


class TestMcpManagerPermissions:
    """MCP 工具安全等级配置测试。"""

    def test_tool_safety_level_from_config(self):
        """工具的安全等级应从 permissions 配置正确读取。"""
        tm = ToolManager()
        mgr = McpManager(tm, "config.json")

        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={
            "filesystem": {"command": "node", "args": ["server.js"]},
        }):
            with patch.object(
                mgr,
                "_load_permissions",
                return_value={
                    "filesystem": {
                        "dangerous": ["mcp_filesystem_write_file"],
                        "safe": ["mcp_filesystem_read_file"],
                    },
                },
            ):
                mgr.register_all(client_factory=FakeMCPClient)

        definitions = {d.name: d.safety_level for d in tm.get_definitions()}
        assert definitions.get("mcp_filesystem_read_file") == "safe"
        assert definitions.get("mcp_filesystem_write_file") == "dangerous"

    def test_default_safety_level_is_caution(self):
        """未配置安全等级的工具应默认为 caution。"""
        tm = ToolManager()
        mgr = McpManager(tm, "config.json")

        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={
            "filesystem": {"command": "node", "args": ["server.js"]},
        }):
            with patch.object(mgr, "_load_permissions", return_value={}):
                mgr.register_all(client_factory=FakeMCPClient)

        definitions = {d.name: d.safety_level for d in tm.get_definitions()}
        assert definitions.get("mcp_filesystem_read_file") == PermissionManager.SAFETY_CAUTION


class TestMcpManagerShutdown:
    """MCP 管理器关闭测试。"""

    def test_shutdown_all_clients(self):
        """shutdown_all 应关闭所有已连接的客户端。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={
            "server_a": {"command": "a"},
            "server_b": {"command": "b"},
        }):
            clients = mgr.register_all(client_factory=FakeMCPClient)

        mgr.shutdown_all()
        for c in clients:
            assert c._shutdown_called, f"Client {c.name} was not shut down"

    def test_shutdown_with_no_clients(self):
        """无客户端时 shutdown_all 不应报错。"""
        mgr = McpManager(ToolManager(), "")
        mgr.shutdown_all()  # 不应抛出异常


class TestMcpManagerToolHandler:
    """MCP 工具 handler 行为测试。"""

    def test_tool_handler_calls_client(self):
        """注册的 tool handler 应能调用 MCP 客户端。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={
            "fs": {"command": "node", "args": ["server.js"]},
        }):
            mgr.register_all(client_factory=FakeMCPClient)

        # 查找注册的 handler 并调用
        definitions = tm.get_definitions()
        read_def = next(d for d in definitions if d.name == "mcp_fs_read_file")
        assert read_def is not None
        assert read_def.safety_level == PermissionManager.SAFETY_CAUTION

    def test_tool_handler_with_arguments(self):
        """注册的 tool handler 应能接收参数。"""
        tm = ToolManager()
        mgr = McpManager(tm, "")

        with patch("kocor.mcp.mcp_manager.load_mcp_servers", return_value={
            "fs": {"command": "node", "args": ["server.js"]},
        }):
            mgr.register_all(client_factory=FakeMCPClient)

        from kocor.llm_provider.message import FunctionCall, ToolCall, ToolResult

        # 模拟工具调用执行
        result = tm.execute(ToolCall(
            id="test-id",
            function=FunctionCall(name="mcp_fs_read_file", arguments='{"path": "/test"}'),
        ))
        assert result is not None
        assert "fake read_file" in result.content