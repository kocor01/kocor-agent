"""测试权限确认钩子"""

import pytest

from kocor.mcp import PermissionManager, PermissionPolicy


class TestPermissionPolicy:
    """测试 PermissionPolicy 数据类"""

    def test_default_values(self):
        policy = PermissionPolicy()
        assert policy.policy == "always_allow"
        assert policy.allowed_tools == []

    def test_custom_values(self):
        policy = PermissionPolicy(
            policy="always_ask",
            allowed_tools=["tool_a", "tool_b"],
        )
        assert policy.policy == "always_ask"
        assert len(policy.allowed_tools) == 2


class TestPermissionManager:
    """测试 PermissionManager"""

    def test_always_allow_by_default(self):
        mgr = PermissionManager({})
        assert mgr.check("any_tool", "any_server") is True

    def test_always_allow_policy(self):
        mgr = PermissionManager({
            "server1": PermissionPolicy(policy="always_allow"),
        })
        assert mgr.check("mcp_server1_anything", "server1") is True

    def test_always_ask_denied_by_input_n(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        mgr = PermissionManager({
            "server1": PermissionPolicy(policy="always_ask"),
        })
        assert mgr.check("mcp_server1_danger", "server1") is False

    def test_always_ask_allowed_by_input_y(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        mgr = PermissionManager({
            "server1": PermissionPolicy(policy="always_ask"),
        })
        assert mgr.check("mcp_server1_danger", "server1") is True

    def test_always_ask_allowed_by_empty_input(self, monkeypatch):
        """默认（回车）视为允许"""
        monkeypatch.setattr("builtins.input", lambda _: "")
        mgr = PermissionManager({
            "server1": PermissionPolicy(policy="always_ask"),
        })
        assert mgr.check("mcp_server1_danger", "server1") is True

    def test_always_ask_session_cache(self, monkeypatch):
        """会话内首次确认后不再重复询问"""
        inputs = iter(["y", "should_not_be_called"])

        def mock_input(_):
            return next(inputs)

        monkeypatch.setattr("builtins.input", mock_input)
        mgr = PermissionManager({
            "server1": PermissionPolicy(policy="always_ask"),
        })

        assert mgr.check("mcp_server1_danger", "server1") is True
        assert mgr.check("mcp_server1_danger", "server1") is True

    def test_allowed_tools_skip_ask(self, monkeypatch):
        """在 allowed_tools 列表中的工具自动放行"""
        monkeypatch.setattr("builtins.input", lambda _: "n")
        mgr = PermissionManager({
            "server1": PermissionPolicy(
                policy="always_ask",
                allowed_tools=["mcp_server1_safe_tool"],
            ),
        })
        assert mgr.check("mcp_server1_safe_tool", "server1") is True
        assert mgr.check("mcp_server1_other", "server1") is False

    def test_multiple_servers_independent(self, monkeypatch):
        """不同服务器的权限策略互相独立"""
        monkeypatch.setattr("builtins.input", lambda _: "n")
        mgr = PermissionManager({
            "safe": PermissionPolicy(policy="always_allow"),
            "risky": PermissionPolicy(policy="always_ask"),
        })
        assert mgr.check("any_tool", "safe") is True
        assert mgr.check("any_tool", "risky") is False

    def test_unknown_server_defaults_to_allow(self):
        """未配置的服务器默认 always_allow"""
        mgr = PermissionManager({})
        assert mgr.check("anything", "unknown") is True
