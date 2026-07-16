"""测试 per-tool 超时覆盖。

TDD Red：验证 ToolDefinition.timeout 字段（None=继承全局 tool_timeout，
0=不超时，正数=自定义秒数）能覆盖 ToolManager.execute 的统一超时。
"""

from __future__ import annotations

import time

from kocor.config import Config
from kocor.llm_provider.message import FunctionCall, ToolCall
from kocor.tools.tool_manager import ToolManager


class TestPerToolTimeout:
    """测试每个工具可独立设置超时，覆盖全局 tool_timeout。"""

    def setup_method(self):
        self._saved_timeout = Config.load().tool_timeout

    def teardown_method(self):
        Config.load().tool_timeout = self._saved_timeout

    def _make_call(self, name: str) -> ToolCall:
        return ToolCall(
            id="call_1",
            function=FunctionCall(name=name, arguments="{}"),
        )

    def test_definition_timeout_defaults_none(self):
        """未指定 timeout 时，ToolDefinition.timeout 为 None（继承全局）。"""
        registry = ToolManager()
        registry.register(
            name="t", description="d", parameters={"type": "object"},
            handler=lambda **kw: "ok",
        )
        defn = registry.get_definitions()[0]
        assert defn.timeout is None

    def test_definition_timeout_zero_stored(self):
        """指定 timeout=0 时，ToolDefinition.timeout 为 0。"""
        registry = ToolManager()
        registry.register(
            name="t", description="d", parameters={"type": "object"},
            handler=lambda **kw: "ok", timeout=0,
        )
        defn = registry.get_definitions()[0]
        assert defn.timeout == 0

    def test_timeout_zero_disables_global_timeout(self):
        """timeout=0 的工具不受全局 tool_timeout 约束（子代理场景）。"""
        Config.load().tool_timeout = 1  # 全局 1s

        registry = ToolManager()

        def slow(**kwargs):
            time.sleep(1.5)  # 超过全局 1s
            return "done"

        registry.register(
            name="slow", description="d", parameters={"type": "object"},
            handler=slow, timeout=0,
        )
        result = registry.execute(self._make_call("slow"))
        assert result.content == "done"

    def test_custom_timeout_overrides_global(self):
        """自定义正数 timeout 覆盖全局（小于全局时按自定义超时）。"""
        Config.load().tool_timeout = 10  # 全局很宽松

        registry = ToolManager()

        def slow(**kwargs):
            time.sleep(3)
            return "done"

        registry.register(
            name="slow", description="d", parameters={"type": "object"},
            handler=slow, timeout=1,  # 但本工具 1s 超时
        )
        result = registry.execute(self._make_call("slow"))
        assert "timeout" in result.content.lower() or "timed out" in result.content.lower()

    def test_default_inherits_global_timeout(self):
        """未指定 timeout 的工具仍受全局 tool_timeout 约束（不回归）。"""
        Config.load().tool_timeout = 1

        registry = ToolManager()

        def slow(**kwargs):
            time.sleep(3)
            return "done"

        registry.register(
            name="slow", description="d", parameters={"type": "object"},
            handler=slow,  # 不传 timeout
        )
        result = registry.execute(self._make_call("slow"))
        assert "timeout" in result.content.lower() or "timed out" in result.content.lower()
