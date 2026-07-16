"""测试 Hook 系统的边缘情况 — AuditLogHook 错误路径、HookManager.register_all。

覆盖代码审查报告指出的「HookManager 和 AuditLogHook 的测试」缺口。
"""

from __future__ import annotations

import json
from unittest.mock import patch

from kocor.hook.base import HookAction, HookContext, HookPoint, HookResult
from kocor.hook.hook_manager import HookManager
from kocor.hook.hooks.audit_log import AuditLogHook
from kocor.logger import Logger

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

        # 验证 POST_GENERATE 点有钩子
        hooks = hm._hooks.get(HookPoint.POST_GENERATE, [])
        assert len(hooks) >= 1
        assert isinstance(hooks[0], AuditLogHook)

    def test_register_all_hook_works(self):
        """register_all 注册的钩子能正常执行。"""
        logger = Logger("INFO")
        hm = HookManager()
        hm.register_all(logger)

        # 执行 POST_GENERATE 钩子，不应报错
        from kocor.llm_provider.message import Message, Usage
        ctx = HookContext(
            iteration=1,
            messages=[],
            response=Message(
                role="assistant",
                content="Hello",
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            ),
        )
        results = hm.run(HookPoint.POST_GENERATE, ctx)

        assert len(results) == 1
        assert results[0].action == HookAction.CONTINUE

    def test_register_all_only_affects_post_generate(self):
        """register_all 只注册到 POST_GENERATE 点。"""
        logger = Logger("INFO")
        hm = HookManager()
        hm.register_all(logger)

        # 其他点不应有钩子
        for point in HookPoint:
            if point != HookPoint.POST_GENERATE:
                assert len(hm._hooks.get(point, [])) == 0, f"{point} should have no hooks"


# ═══════════════════════════════════════════════
# AuditLogHook 错误路径
# ═══════════════════════════════════════════════


class TestAuditLogHookErrorPaths:
    """AuditLogHook 错误路径测试。"""

    def setup_method(self):
        self._logger = Logger("INFO")

    def test_run_without_response(self):
        """无 response 时日志标记 usage 为 unavailable。"""
        hook = AuditLogHook(logger=self._logger)
        with patch.object(hook._logger, 'audit') as mock_audit:
            ctx = HookContext(iteration=1, messages=[])
            hook.run(ctx)

            call_args = mock_audit.call_args[0][0]
            entry = json.loads(call_args)
            assert entry["usage"] == "unavailable"
            assert entry["iteration"] == 1

    def test_run_without_usage(self):
        """response 无 usage 时日志标记 usage 为 unavailable。"""
        hook = AuditLogHook(logger=self._logger)
        with patch.object(hook._logger, 'audit') as mock_audit:
            from kocor.llm_provider.message import Message
            ctx = HookContext(
                iteration=1,
                messages=[],
                response=Message(role="assistant", content="No usage"),
            )
            hook.run(ctx)

            call_args = mock_audit.call_args[0][0]
            entry = json.loads(call_args)
            assert entry["usage"] == "unavailable"

    def test_run_with_partial_usage(self):
        """部分 usage 字段为 0 也能正确记录。"""
        hook = AuditLogHook(logger=self._logger)
        with patch.object(hook._logger, 'audit') as mock_audit:
            from kocor.llm_provider.message import Message, Usage
            ctx = HookContext(
                iteration=1,
                messages=[],
                response=Message(
                    role="assistant",
                    content="Partial",
                    usage=Usage(prompt_tokens=100, completion_tokens=0, total_tokens=100, cached_tokens=0),
                ),
            )
            hook.run(ctx)

            call_args = mock_audit.call_args[0][0]
            entry = json.loads(call_args)
            assert entry["prompt_tokens"] == 100
            assert entry["completion_tokens"] == 0
            assert entry["total_tokens"] == 100
            assert entry["cached_tokens"] == 0


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
                return HookResult(action=HookAction.ABORT, message="skip")

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
        assert HookAction.ABORT == "abort"