"""钩子系统测试。"""

from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction
from kocor.hook.hook_manager import HookManager
from kocor.hook.hooks import AuditLogHook


class TestHookPoint:
    def test_enum_values(self):
        assert HookPoint.PRE_GENERATE.value == "pre_generate"
        assert HookPoint.POST_GENERATE.value == "post_generate"
        assert HookPoint.PRE_TOOL.value == "pre_tool"
        assert HookPoint.POST_TOOL.value == "post_tool"
        assert HookPoint.ON_ERROR.value == "on_error"
        assert HookPoint.ON_BUDGET_EXHAUSTED.value == "on_budget_exhausted"


class TestHookManager:
    def test_register_and_run(self):
        runner = HookManager()
        results = []

        class TestHook:
            hook_point = HookPoint.PRE_TOOL

            def run(self, ctx):
                results.append("called")
                return HookResult(action=HookAction.CONTINUE)

        runner.register(TestHook())
        runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert results == ["called"]

    def test_multiple_hooks_same_point(self):
        runner = HookManager()
        order = []

        class HookA:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                order.append("a")
                return HookResult(action=HookAction.CONTINUE)

        class HookB:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                order.append("b")
                return HookResult(action=HookAction.CONTINUE)

        runner.register(HookA())
        runner.register(HookB())
        runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert order == ["a", "b"]

    def test_abort_stops_remaining_hooks(self):
        runner = HookManager()
        order = []

        class Aborter:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                order.append("abort")
                return HookResult(action=HookAction.ABORT)

        class AfterAbort:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                order.append("after")
                return HookResult(action=HookAction.CONTINUE)

        runner.register(Aborter())
        runner.register(AfterAbort())
        runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert order == ["abort"]

    def test_hook_exception_does_not_break(self):
        runner = HookManager()

        class BrokenHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                raise RuntimeError("boom")

        class GoodHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.CONTINUE)

        runner.register(BrokenHook())
        runner.register(GoodHook())
        results = runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert len(results) == 2
        assert results[0].action == HookAction.CONTINUE
        assert "boom" in results[0].message
        assert results[1].action == HookAction.CONTINUE

    def test_no_hooks_at_point(self):
        runner = HookManager()
        results = runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert results == []

    def test_different_points_isolated(self):
        runner = HookManager()
        pre_results = []
        post_results = []

        class PreHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                pre_results.append("pre")
                return HookResult(action=HookAction.CONTINUE)

        class PostHook:
            hook_point = HookPoint.POST_TOOL
            def run(self, ctx):
                post_results.append("post")
                return HookResult(action=HookAction.CONTINUE)

        runner.register(PreHook())
        runner.register(PostHook())
        runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert pre_results == ["pre"]
        assert post_results == []

    def test_skip_tool_action(self):
        runner = HookManager()

        class Skipper:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                return HookResult(action=HookAction.SKIP_TOOL, message="skip it")

        runner.register(Skipper())
        results = runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert results[0].action == HookAction.SKIP_TOOL
        assert results[0].message == "skip it"

    def test_clear_all_hooks(self):
        runner = HookManager()
        tracker = []

        class SomeHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                tracker.append("called")
                return HookResult(action=HookAction.CONTINUE)

        runner.register(SomeHook())
        runner.clear()
        runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert tracker == []

    def test_unregister(self):
        runner = HookManager()
        tracker = []

        class SomeHook:
            hook_point = HookPoint.PRE_TOOL
            def run(self, ctx):
                tracker.append("called")
                return HookResult(action=HookAction.CONTINUE)

        hook = SomeHook()
        runner.register(hook)
        runner.unregister(hook)
        runner.run(HookPoint.PRE_TOOL, HookContext(iteration=1, messages=[]))
        assert tracker == []


class TestAuditLogHook:
    def test_hook_point(self):
        hook = AuditLogHook()
        assert hook.hook_point == HookPoint.POST_TOOL

    def test_run_returns_continue(self, tmp_path):
        log_file = tmp_path / "test_harness.log"
        hook = AuditLogHook(log_path=str(log_file))
        from kocor.llm_provider.message import ToolCall, FunctionCall

        result = hook.run(HookContext(
            iteration=1,
            messages=[],
            tool_call=ToolCall(
                id="call_1",
                function=FunctionCall(name="read_file", arguments='{"path": "test.txt"}'),
            ),
        ))
        assert result.action == HookAction.CONTINUE
        log_content = log_file.read_text()
        assert "read_file" in log_content
        assert "call_1" in log_content

    def test_run_without_tool_call(self, tmp_path):
        hook = AuditLogHook(log_path=str(tmp_path / "empty.log"))
        result = hook.run(HookContext(iteration=1, messages=[]))
        assert result.action == HookAction.CONTINUE