"""测试 MCP 客户端（基于官方 SDK）"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from mcp.types import CallToolResult, InitializeResult, ListToolsResult, TextContent, Tool

from kocor.llm_provider.message import FunctionCall, ToolCall
from kocor.tools.tool_manager import ToolManager

# ── 测试辅助函数 ──────────────────────────────────────────────────────────

def _mock_tool(name: str, description: str = "", inputSchema: dict | None = None) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=inputSchema or {"type": "object"},
    )


def _make_async_cm(return_value):
    """创建模拟的异步上下文管理器。"""
    cm = AsyncMock()
    cm.__aenter__.return_value = return_value
    return cm


# ── SanitizeServerName ────────────────────────────────────────────────────


class TestSanitizeServerName:
    def test_lowercase(self):
        from kocor.mcp import sanitize_server_name
        assert sanitize_server_name("GitHub") == "github"
        assert sanitize_server_name("MyServer") == "myserver"

    def test_special_chars_to_underscore(self):
        from kocor.mcp import sanitize_server_name
        assert sanitize_server_name("my-server") == "my_server"
        assert sanitize_server_name("foo.bar") == "foo_bar"

    def test_alphanumeric_unchanged(self):
        from kocor.mcp import sanitize_server_name
        assert sanitize_server_name("filesystem") == "filesystem"


# ── MCPConfig ─────────────────────────────────────────────────────────────


class TestMCPConfig:
    def test_default_values(self):
        from kocor.mcp import MCPConfig
        cfg = MCPConfig(command="node")
        assert cfg.command == "node"
        assert cfg.args == []
        assert cfg.env == {}

    def test_remote_config(self):
        from kocor.mcp import MCPConfig
        cfg = MCPConfig(url="https://example.com/mcp", transport="sse")
        assert cfg.url == "https://example.com/mcp"
        assert cfg.transport == "sse"


# ── LoadMCPServers ────────────────────────────────────────────────────────


class TestLoadMCPServers:
    def test_load_valid_config(self):
        from kocor.mcp import load_mcp_servers

        data = json.dumps({
            "mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "fs"]},
                "api": {"url": "https://example.com/mcp"},
            }
        })
        with patch("kocor.mcp.config.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=data)):
            servers = load_mcp_servers("cfg.json")

        assert len(servers) == 2
        assert servers["fs"].command == "npx"
        assert servers["api"].url == "https://example.com/mcp"

    def test_file_not_found(self):
        from kocor.mcp import load_mcp_servers
        assert load_mcp_servers("/nonexistent") == {}

    def test_invalid_json(self):
        from kocor.mcp import load_mcp_servers
        with patch("kocor.mcp.config.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="bad")):
            assert load_mcp_servers("bad.json") == {}


# ── MCPClient ──────────────────────────────────────────────────────────────


class TestMCPClient:
    """测试基于 SDK 的 MCPClient。"""

    def _setup_sdk_mocks(self, mock_stdio, mock_session_cls,
                         init_result=None, tools=(), call_result=None):
        """统一设置 SDK mock 环境。"""
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value = _make_async_cm((mock_read, mock_write))

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.initialize = AsyncMock(return_value=init_result or InitializeResult(
            protocolVersion="2025-03-26",
            capabilities={"tools": {}},
            serverInfo={"name": "test", "version": "1.0"},
        ))
        mock_session.list_tools = AsyncMock(return_value=ListToolsResult(
            tools=tools if tools is not None else [_mock_tool("echo")],
        ))
        mock_session.call_tool = AsyncMock(return_value=call_result or CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        ))
        mock_session_cls.return_value = mock_session
        return mock_session

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_initialize_success(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        self._setup_sdk_mocks(mock_stdio, mock_session_cls)

        client = MCPClient("test", MCPConfig(command="python", args=["-m", "server"]))
        client.start()

        # verify initialize was called
        mock_session_cls.assert_called_once()
        assert client._session is not None

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_initialize_protocol_version_too_old(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig, MCPError

        self._setup_sdk_mocks(
            mock_stdio, mock_session_cls,
            init_result=InitializeResult(
                protocolVersion="2024-10-01",
                capabilities={},
                serverInfo={"name": "old", "version": "1.0"},
            ),
        )

        client = MCPClient("test", MCPConfig(command="old_server"))
        with pytest.raises(MCPError, match="protocol version"):
            client.start()

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_list_tools(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        mock_sess = self._setup_sdk_mocks(
            mock_stdio, mock_session_cls,
            tools=[
                _mock_tool("read", "Read file", {
                    "type": "object", "properties": {"path": {"type": "string"}},
                }),
                _mock_tool("write", "Write file", {
                    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                }),
            ],
        )

        client = MCPClient("test", MCPConfig(command="server"))
        client.start()
        tools = client.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "read"
        assert tools[1]["name"] == "write"
        assert tools[0]["inputSchema"]["properties"]["path"]["type"] == "string"

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_list_tools_empty(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        self._setup_sdk_mocks(mock_stdio, mock_session_cls, tools=[])

        client = MCPClient("test", MCPConfig(command="server"))
        client.start()
        assert client.list_tools() == []

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_call_tool_success(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        self._setup_sdk_mocks(
            mock_stdio, mock_session_cls,
            call_result=CallToolResult(
                content=[TextContent(type="text", text="hello world")],
                isError=False,
            ),
        )

        client = MCPClient("test", MCPConfig(command="server"))
        client.start()
        result = client.call_tool("echo", {"msg": "hi"})

        assert result == "hello world"

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_call_tool_error(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        self._setup_sdk_mocks(
            mock_stdio, mock_session_cls,
            call_result=CallToolResult(
                content=[TextContent(type="text", text="fail message")],
                isError=True,
            ),
        )

        client = MCPClient("test", MCPConfig(command="server"))
        client.start()
        result = client.call_tool("bad", {})

        assert "[MCP Error]" in result
        assert "fail message" in result

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_call_tool_multiple_content_blocks(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        self._setup_sdk_mocks(
            mock_stdio, mock_session_cls,
            call_result=CallToolResult(
                content=[
                    TextContent(type="text", text="part1"),
                    TextContent(type="text", text="part2"),
                ],
                isError=False,
            ),
        )

        client = MCPClient("test", MCPConfig(command="server"))
        client.start()
        result = client.call_tool("multi", {})

        assert "part1" in result
        assert "part2" in result

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_shutdown(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        self._setup_sdk_mocks(mock_stdio, mock_session_cls)

        client = MCPClient("test", MCPConfig(command="server"))
        client.start()
        client.shutdown()

        # session.__aexit__ should be called
        assert client._session is None

    def test_shutdown_not_started(self):
        from kocor.mcp import MCPClient, MCPConfig

        client = MCPClient("test", MCPConfig(command="server"))
        client.shutdown()  # should not raise

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.stdio.stdio_client")
    def test_empty_command_raises(self, mock_stdio, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig, MCPError

        client = MCPClient("test", MCPConfig(command=""))
        with pytest.raises(MCPError, match="empty command"):
            client.start()

    # ── HTTP 传输 ─────────────────────────────────────────────────────────

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.streamable_http.streamable_http_client")
    def test_http_transport(self, mock_http, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_http.return_value = _make_async_cm((mock_read, mock_write))

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.initialize = AsyncMock(return_value=InitializeResult(
            protocolVersion="2025-03-26", capabilities={},
            serverInfo={"name": "remote", "version": "1.0"},
        ))
        mock_session.list_tools = AsyncMock(return_value=ListToolsResult(
            tools=[_mock_tool("hello")],
        ))
        mock_session.call_tool = AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="pong")], isError=False,
        ))
        mock_session_cls.return_value = mock_session

        client = MCPClient("remote", MCPConfig(url="https://example.com/mcp"))
        client.start()
        tools = client.list_tools()
        result = client.call_tool("hello", {})

        assert len(tools) == 1
        assert tools[0]["name"] == "hello"
        assert result == "pong"

    @patch("kocor.mcp.client.ClientSession")
    @patch("mcp.client.streamable_http.streamable_http_client")
    def test_http_transport_error(self, mock_http, mock_session_cls):
        from kocor.mcp import MCPClient, MCPConfig

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_http.return_value = _make_async_cm((mock_read, mock_write))

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_session.initialize = AsyncMock(return_value=InitializeResult(
            protocolVersion="2025-03-26", capabilities={},
            serverInfo={"name": "remote", "version": "1.0"},
        ))
        # Simulate connection failure on call_tool
        mock_session.call_tool = AsyncMock(side_effect=Exception("Connection lost"))
        mock_session_cls.return_value = mock_session

        client = MCPClient("remote", MCPConfig(url="https://example.com/mcp"))
        client.start()

        with pytest.raises(Exception, match="Connection lost"):
            client.call_tool("bad", {})


# ── RegisterMCPTools ──────────────────────────────────────────────────────


class TestRegisterMCPTools:
    def test_stdio_tools_registered_with_mcp_prefix(self):
        from kocor.mcp import McpManager

        config_data = json.dumps({
            "mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "fs"]},
            }
        })

        with patch("mcp.client.stdio.stdio_client") as mock_stdio, \
             patch("kocor.mcp.client.ClientSession") as mock_session_cls, \
             patch("kocor.mcp.config.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=config_data)):

            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_stdio.return_value = _make_async_cm((mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock(return_value=InitializeResult(
                protocolVersion="2025-03-26", capabilities={},
                serverInfo={"name": "fs", "version": "1.0"},
            ))
            mock_session.list_tools = AsyncMock(return_value=ListToolsResult(
                tools=[_mock_tool("read", "Read file", {
                    "type": "object", "properties": {"path": {"type": "string"}},
                })],
            ))
            mock_session_cls.return_value = mock_session

            registry = ToolManager()
            manager = McpManager(registry, "mcp.json")
            clients = manager.register_all()

        assert len(clients) == 1
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].name == "mcp_fs_read"
        assert defs[0].description == "Read file"

    def test_one_server_fails_others_still_register(self):
        from kocor.mcp import McpManager

        config_data = json.dumps({
            "mcpServers": {
                "good": {"command": "good-server"},
                "bad": {"command": "bad-server"},
            }
        })

        call_count = [0]

        def mock_session_side(*args, **kwargs):
            call_count[0] += 1
            mock_s = AsyncMock()
            mock_s.__aenter__.return_value = mock_s
            if call_count[0] == 2:
                mock_s.initialize = AsyncMock(side_effect=Exception("Connection refused"))
            else:
                mock_s.initialize = AsyncMock(return_value=InitializeResult(
                    protocolVersion="2025-03-26", capabilities={},
                    serverInfo={"name": "g", "version": "1.0"},
                ))
                mock_s.list_tools = AsyncMock(return_value=ListToolsResult(
                    tools=[_mock_tool("tool1")]
                ))
            return mock_s

        with patch("mcp.client.stdio.stdio_client") as mock_stdio, \
             patch("kocor.mcp.client.ClientSession", side_effect=mock_session_side), \
             patch("kocor.mcp.config.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=config_data)):

            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_read.__aenter__.return_value = mock_read
            mock_write.__aenter__.return_value = mock_write
            mock_stdio.return_value = _make_async_cm((mock_read, mock_write))

            registry = ToolManager()
            manager = McpManager(registry, "mcp.json")
            clients = manager.register_all()

        assert len(clients) == 1
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].name == "mcp_good_tool1"

    def test_execute_mcp_tool_via_registry(self):
        from kocor.mcp import McpManager

        config_data = json.dumps({
            "mcpServers": {
                "demo": {"command": "demo-server"},
            }
        })

        with patch("mcp.client.stdio.stdio_client") as mock_stdio, \
             patch("kocor.mcp.client.ClientSession") as mock_session_cls, \
             patch("kocor.mcp.config.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=config_data)):

            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_stdio.return_value = _make_async_cm((mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock(return_value=InitializeResult(
                protocolVersion="2025-03-26", capabilities={},
                serverInfo={"name": "demo", "version": "1.0"},
            ))
            mock_session.list_tools = AsyncMock(return_value=ListToolsResult(
                tools=[_mock_tool("echo", inputSchema={
                    "type": "object", "properties": {"msg": {"type": "string"}},
                })],
            ))
            mock_session.call_tool = AsyncMock(return_value=CallToolResult(
                content=[TextContent(type="text", text="Hello from MCP")],
                isError=False,
            ))
            mock_session_cls.return_value = mock_session

            registry = ToolManager()
            manager = McpManager(registry, "mcp.json")
            clients = manager.register_all()

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="mcp_demo_echo", arguments='{"msg": "hello"}'),
        )
        result = registry.execute(tool_call)

        assert result.content == "Hello from MCP"

        manager.shutdown_all()


# ── McpManager ──────────────────────────────────────────────────────────────


class TestMcpManager:
    def test_shutdown_all_calls_client_shutdown(self):
        from kocor.mcp import McpManager

        manager = McpManager(ToolManager())
        c1 = MagicMock()
        c2 = MagicMock()
        manager._clients = [c1, c2]
        manager.shutdown_all()
        c1.shutdown.assert_called_once()
        c2.shutdown.assert_called_once()

    def test_shutdown_all_with_one_failure(self):
        from kocor.mcp import McpManager

        manager = McpManager(ToolManager())
        c1 = MagicMock()
        c1.shutdown.side_effect = Exception("fail")
        c2 = MagicMock()
        manager._clients = [c1, c2]
        manager.shutdown_all()  # should not raise
        c2.shutdown.assert_called_once()

    def test_shutdown_all_empty(self):
        from kocor.mcp import McpManager

        manager = McpManager(ToolManager())
        manager._clients = []
        manager.shutdown_all()  # should not raise