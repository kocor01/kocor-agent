"""测试子代理非交互权限策略。

TDD Red：验证 POLICY_NONINTERACTIVE 下 safe/caution 放行、
dangerous 按 subagent_auto_approve 决定、永不调用 input()。
"""

from __future__ import annotations

from kocor.config import Config
from kocor.llm_provider.message import FunctionCall, ToolCall
from kocor.tools.permission import PermissionManager
from kocor.tools.tool_manager import ToolManager


class TestNonInteractivePermission:
    """测试非交互权限策略。"""

    def setup_method(self):
        self._saved_auto_approve = Config.load().subagent_auto_approve
        # 构造一个带安全等级的 ToolManager
        self.tm = ToolManager()
        self.tm.register(
            name="safe_tool", description="s", parameters={"type": "object"},
            handler=lambda **kw: "ok", safety_level=PermissionManager.SAFETY_SAFE,
        )
        self.tm.register(
            name="caution_tool", description="c", parameters={"type": "object"},
            handler=lambda **kw: "ok", safety_level=PermissionManager.SAFETY_CAUTION,
        )
        self.tm.register(
            name="dangerous_tool", description="d", parameters={"type": "object"},
            handler=lambda **kw: "ok", safety_level=PermissionManager.SAFETY_DANGEROUS,
        )

    def teardown_method(self):
        Config.load().subagent_auto_approve = self._saved_auto_approve

    def _make_call(self, name: str) -> ToolCall:
        return ToolCall(id="call_1", function=FunctionCall(name=name, arguments="{}"))

    # --- 默认拒绝危险 ---

    def test_noninteractive_allows_safe(self):
        pm = PermissionManager(
            policy=PermissionManager.POLICY_NONINTERACTIVE,
            tool_manager=self.tm,
        )
        assert pm.check(self._make_call("safe_tool")) is True

    def test_noninteractive_allows_caution(self):
        pm = PermissionManager(
            policy=PermissionManager.POLICY_NONINTERACTIVE,
            tool_manager=self.tm,
        )
        assert pm.check(self._make_call("caution_tool")) is True

    def test_noninteractive_denies_dangerous_by_default(self):
        """auto_approve=False（默认）时，dangerous 工具被拒绝。"""
        Config.load().subagent_auto_approve = False
        pm = PermissionManager(
            policy=PermissionManager.POLICY_NONINTERACTIVE,
            tool_manager=self.tm,
        )
        assert pm.check(self._make_call("dangerous_tool")) is False

    def test_noninteractive_auto_approve_allows_dangerous(self):
        """auto_approve=True 时，dangerous 工具被放行。"""
        Config.load().subagent_auto_approve = True
        pm = PermissionManager(
            policy=PermissionManager.POLICY_NONINTERACTIVE,
            tool_manager=self.tm,
        )
        assert pm.check(self._make_call("dangerous_tool")) is True

    # --- 永不调用 input() ---

    def test_noninteractive_never_calls_input(self, monkeypatch):
        """非交互策略绝不调用 input()（monkeypatch 抛错，确保不被调用）。"""
        def bomb(*args, **kwargs):
            raise RuntimeError("input() should never be called in noninteractive mode")

        monkeypatch.setattr("builtins.input", bomb)

        Config.load().subagent_auto_approve = False
        pm = PermissionManager(
            policy=PermissionManager.POLICY_NONINTERACTIVE,
            tool_manager=self.tm,
        )
        # safe 和 caution 直接放行不调 input；dangerous 直接拒绝也不调
        assert pm.check(self._make_call("safe_tool")) is True
        assert pm.check(self._make_call("caution_tool")) is True
        assert pm.check(self._make_call("dangerous_tool")) is False

        # auto_approve=True 时 dangerous 也放行，同样不调 input
        Config.load().subagent_auto_approve = True
        assert pm.check(self._make_call("dangerous_tool")) is True