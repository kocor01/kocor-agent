"""测试 Hook 系统的边缘情况 — AuditLogHook 错误路径、HookManager.register_all。

覆盖代码审查报告指出的「HookManager 和 AuditLogHook 的测试」缺口。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kocor.logger import Logger
from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction
from kocor.hook.hook_manager import HookManager
from kocor.hook.hooks.audit_log import AuditLogHook
from kocor.llm_provider.message import FunctionCall, ToolCall


# ═══════════════════════════════════════════════
# HookManager.register_all
# ═══════════════════════════════════════════════


class TestHookManagerRegisterAll:
    """HookManager.register_all 集成测试。"""

    def test_register_all_creates_audit_log_hook(self):
        """register_all 注册 AuditLogHook。"""
        logger = Logger("INFO")
        hm = HookManager()
        hm.register_all(logger)

        # 验证 POST_TOOL 点有钩子
        hooks = hm._hooks.get(HookPoint.POST_TOOL, [])
        assert len(hooks) >= 1
        assert isinstance(hooks[0], AuditLogHook)

    def test_register_all_hook_works(self):
        """register_all 注册的钩子能正常执行。"""
        logger = Logger("INFO")
        hm = HookManager()
        hm.register_all(logger)

        # 执行 POST_TOOL 钩子，不应报错
        ctx = HookContext(
            iteration=1,
            messages=[],
            tool_call=ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
        )
        results = hm.run(HookPoint.POST_TOOL, ctx)

        assert len(results) == 1
        assert results[0].action == HookAction.CONTINUE

    def test_register_all_only_affects_post_tool(self):
        """register_all 只注册到 POST_TOOL 点。"""
        logger = Logger("INFO")
        hm = HookManager()
        hm.register_all(logger)

        # 其他点不应有钩子
        for point in HookPoint:
            if point != HookPoint.POST_TOOL:
                assert len(hm._hooks.get(point, [])) == 0, f"{point} should have no hooks"


# ═══════════════════════════════════════════════
# AuditLogHook 错误路径
# ═══════════════════════════════════════════════


class TestAuditLogHookErrorPaths:
    """AuditLogHook 错误路径测试。"""

    def setup_method(self):
        self.logger = Logger("INFO")

    def test_run_with_error_context(self):
        """包含 error 上下文的审计日志。"""
        hook = AuditLogHook(logger=self.logger)
        with patch.object(self.logger, 'info') as mock_info:
            ctx = HookContext(
                iteration=1,
                messages=[],
                tool_call=ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
                error=RuntimeError("file not found"),
            )
            hook.run(ctx)

            # 验证日志内容包含错误信息
            call_args = mock_info.call_args[0][0]
            entry = json.loads(call_args)
            assert entry["error"] == "file not found"
            assert entry["tool"] == "read_file"

    def test_run_without_tool_call(self):
        """无 tool_call 时日志不包含工具信息。"""
        hook = AuditLogHook(logger=self.logger)
        with patch.object(self.logger, 'info') as mock_info:
            ctx = HookContext(iteration=1, messages=[])
            hook.run(ctx)

            call_args = mock_info.call_args[0][0]
            entry = json.loads(call_args)
            assert "tool" not in entry
            assert "arguments" not in entry
            assert "tool_call_id" not in entry
            assert entry["iteration"] == 1

    def test_run_without_error(self):
        """无 error 时日志不包含 error 字段。"""
        hook = AuditLogHook(logger=self.logger)
        with patch.object(self.logger, 'info') as mock_info:
            ctx = HookContext(
                iteration=1,
                messages=[],
                tool_call=ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
            )
            hook.run(ctx)

            call_args = mock_info.call_args[0][0]
            entry = json.loads(call_args)
            assert "error" not in entry
            assert entry["tool"] == "read_file"
            assert entry["tool_call_id"] == "c1"

    def test_run_with_unicode_arguments(self):
        """Unicode 参数在日志中正确处理。"""
        hook = AuditLogHook(logger=self.logger)
        with patch.object(self.logger, 'info') as mock_info:
            ctx = HookContext(
                iteration=1,
                messages=[],
                tool_call=ToolCall(id="c1", function=FunctionCall(name="search", arguments='{"q": "中文测试"}')),
            )
            hook.run(ctx)

            call_args = mock_info.call_args[0][0]
            entry = json.loads(call_args)
            assert entry["arguments"] == '{"q": "中文测试"}'


# ═══════════════════════════════════════════════
# HookManager 多重注册和冲突
# ═══════════════════════════════════════════════


class TestHookManagerMultiregister:
    """HookManager 多重注册和冲突场景。"""

    def test_register_same_hook_twice(self):
        """同一钩子实例注册两次不影响。"""
        hm = HookManager()

        class TestHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.CONTINUE)

        hook = TestHook()
        hm.register(hook)
        hm.register(hook)  # 第二次注册

        hooks = hm._hooks[HookPoint.PRE_TOOL]
        assert len(hooks) == 2  # 允许重复注册

    def test_run_with_mixed_actions(self):
        """混合不同 action 的钩子。"""
        hm = HookManager()

        class ContinueHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.CONTINUE)

        class SkipHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.SKIP_TOOL, message="skip")

        class AbortHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.ABORT)

        hm.register(ContinueHook())
        hm.register(AbortHook())  # abort 后停止
        hm.register(SkipHook())  # 不应执行

        results = hm.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert len(results) == 2
        assert results[0].action == HookAction.CONTINUE
        assert results[1].action == HookAction.ABORT

    def test_clear_after_register_all(self):
        """register_all 后 clear 清除所有钩子。"""
        logger = Logger("INFO")
        hm = HookManager()
        hm.register_all(logger)
        hm.clear()

        assert len(hm._hooks) == 0

    def test_unregister_nonexistent_hook(self):
        """移除不存在的钩子不报错。"""
        hm = HookManager()

        class TestHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.CONTINUE)

        hook = TestHook()
        hm.unregister(hook)  # 不应报错

    def test_empty_hook_point_still_returns_list(self):
        """无钩子的 HookPoint 返回空列表。"""
        hm = HookManager()
        results = hm.run(HookPoint.PRE_SUMMARIZE, HookContext(iteration=1, messages=[]))
        assert results == []


# ═══════════════════════════════════════════════
# HookContext 创建
# ═══════════════════════════════════════════════


class TestHookContextCreation:
    """HookContext 创建和默认值。"""

    def test_minimal_context(self):
        """最小上下文。"""
        ctx = HookContext(iteration=1, messages=[])
        assert ctx.iteration == 1
        assert ctx.messages == []
        assert ctx.tool_call is None
        assert ctx.tool_result is None
        assert ctx.error is None
        assert ctx.config == {}

    def test_full_context(self):
        """完整上下文。"""
        ctx = HookContext(
            iteration=2,
            messages=["msg1"],
            tool_call="tc",
            tool_result="tr",
            error=ValueError("bad"),
            config={"key": "val"},
        )
        assert ctx.iteration == 2
        assert ctx.tool_call == "tc"
        assert ctx.tool_result == "tr"
        assert isinstance(ctx.error, ValueError)
        assert ctx.config["key"] == "val"


# ═══════════════════════════════════════════════
# HookAction 枚举
# ═══════════════════════════════════════════════


class TestHookActionEnum:
    """HookAction 枚举值。"""

    def test_action_values(self):
        assert HookAction.CONTINUE == "continue"
        assert HookAction.SKIP_TOOL == "skip_tool"
        assert HookAction.ABORT == "abort"