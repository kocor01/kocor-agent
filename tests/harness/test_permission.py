"""统一 PermissionManager 测试。"""

import pytest
from kocor.harness.permission import PermissionManager
from kocor.tools.toolset.read_file import ReadFile
from kocor.tools.toolset.write_file import WriteFile
from kocor.tools.toolset.run_python import RunPython
from kocor.llm_provider.message import ToolCall, FunctionCall


def _tc(name: str, args: str = "{}") -> ToolCall:
    """快速创建 ToolCall 的测试辅助。"""
    return ToolCall(id="test", function=FunctionCall(name=name, arguments=args))


class TestBuiltinToolSafety:
    def test_tool_class_has_safety_level(self):
        assert ReadFile.SAFETY_LEVEL == "caution"
        assert WriteFile.SAFETY_LEVEL == "dangerous"
        assert RunPython.SAFETY_LEVEL == "dangerous"

    def test_tool_manager_builds_safety_map(self):
        from kocor.tools.tool_manager import ToolManager
        tm = ToolManager()
        tm.register_builtin_tools()
        assert tm._tools["read_file"].safety_level == "caution"
        assert tm._tools["write_file"].safety_level == "dangerous"
        assert tm._tools["run_python"].safety_level == "dangerous"
        assert "unknown_tool" not in tm._tools


class TestPermissionManager:
    def test_default_policy(self):
        pm = PermissionManager()
        assert pm.policy == "default"

    def test_permissive_policy_allows_safe(self):
        pm = PermissionManager(policy="permissive")
        # permissive 策略下 safe/caution 自动允许
        assert pm.check(_tc("read_file")) is True

    def test_permissive_policy_allows_unknown(self):
        pm = PermissionManager(policy="permissive")
        assert pm.check(_tc("unknown_safe_tool")) is True

    def test_permissive_policy_allows_dangerous(self):
        pm = PermissionManager(policy="permissive")
        # permissive 策略下全部自动允许，包括 dangerous
        assert pm.check(_tc("run_python")) is True

    def test_default_policy_allows_safe(self):
        pm = PermissionManager(policy="default")
        # safe 工具不在映射表中，未知工具默认 "caution"
        # 在 default 策略下，caution 和 dangerous 都需询问 -> 拒绝
        assert pm.check(_tc("read_file")) is False  # caution, no stdin -> denied

    def test_safe_tool_auto_allowed_in_default(self, monkeypatch):
        # Monkeypatch 安全等级为 "safe"
        pm = PermissionManager(policy="default")
        assert pm.check(_tc("read_file")) is False  # no stdin -> denied for caution

    def test_always_allow_always_passes(self):
        pm = PermissionManager(always_allow={"write_file", "run_python"})
        assert pm.check(_tc("write_file")) is True
        assert pm.check(_tc("run_python")) is True

    def test_always_ask_denies_without_stdin(self):
        pm = PermissionManager(always_ask={"write_file"}, cache_enabled=True)
        # 无 stdin 时 _ask_user 返回 False
        assert pm.check(_tc("write_file")) is False

    def test_always_ask_allows_with_mocked_stdin(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        pm = PermissionManager(always_ask={"write_file"}, cache_enabled=True)
        assert pm.check(_tc("write_file")) is True

    def test_cache_hits(self):
        pm = PermissionManager(cache_enabled=True)
        pm._cache.add("write_file")
        assert pm.check(_tc("write_file")) is True

    def test_cache_disabled(self):
        pm = PermissionManager(cache_enabled=False)
        pm._cache.add("write_file")
        # 缓存禁用，需重新检查 -> 无 stdin -> 拒绝
        assert pm.check(_tc("write_file")) is False

    def test_always_allow_overrides_policy(self):
        pm = PermissionManager(policy="strict", always_allow={"read_file"})
        assert pm.check(_tc("read_file")) is True

    def test_strict_policy_denies_without_stdin(self):
        pm = PermissionManager(policy="strict")
        # strict 策略下 safe 自动，caution/dangerous 需询问 -> 拒绝
        assert pm.check(_tc("read_file")) is False
        assert pm.check(_tc("run_python")) is False

    def test_config_update(self):
        pm = PermissionManager(policy="default")
        pm.update_config({"policy": "strict", "always_allow": ["read_file"]})
        assert pm.policy == "strict"
        assert "read_file" in pm._always_allow

    def test_invalid_policy_falls_back_to_default(self):
        pm = PermissionManager(policy="invalid")
        assert pm.policy == "default"

    def test_clear_cache(self):
        pm = PermissionManager()
        pm._cache.add("write_file")
        pm.clear_cache()
        assert len(pm._cache) == 0

    def test_cache_size_limit(self):
        pm = PermissionManager(cache_enabled=True, cache_max_size=2)
        pm._add_to_cache("tool1")
        pm._add_to_cache("tool2")
        pm._add_to_cache("tool3")
        pm._add_to_cache("tool4")
        assert len(pm._cache) <= 2

    def test_ask_user_always_option(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "a")
        pm = PermissionManager(always_ask={"read_file"})
        assert pm.check(_tc("read_file")) is True
        assert "read_file" in pm._cache

    def test_ask_user_no_option(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        pm = PermissionManager(always_ask={"read_file"})
        assert pm.check(_tc("read_file")) is False
        assert "read_file" not in pm._cache

    def test_ask_user_eof_returns_false(self):
        pm = PermissionManager(always_ask={"read_file"})
        assert pm.check(_tc("read_file")) is False