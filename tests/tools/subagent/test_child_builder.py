"""测试子代理构建器（child_builder）。

TDD：验证工具集收窄、消息 seed、角色判定。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from kocor.config import Config
from kocor.tools.toolsets.subagent.child_builder import (
    _build_child_tool_manager,
    assemble_child_loop,
)


class MockLLM:
    """最小化 LLM 客户端 mock，仅满足类型检查。"""
    def generate(self, messages, **kwargs):
        return MagicMock()
    def stream(self, messages, **kwargs):
        return iter([])


class TestBuildChildToolManager:
    """测试 _build_child_tool_manager 工具集收窄。"""

    def test_leaf_strips_subagent(self):
        """leaf 角色（include_subagent=False）不含 subagent 工具。"""
        tm = _build_child_tool_manager(
            blocked_tools=("memory",),
            include_subagent=False,
        )
        names = [d.name for d in tm.get_definitions()]
        assert "subagent" not in names

    def test_orchestrator_keeps_subagent(self):
        """orchestrator 角色（include_subagent=True）保留 subagent 工具。"""
        tm = _build_child_tool_manager(
            blocked_tools=("memory",),
            include_subagent=True,
        )
        names = [d.name for d in tm.get_definitions()]
        assert "subagent" in names

    def test_blocks_configured_tools(self):
        """blocked_tools 中的工具被剥离。"""
        tm = _build_child_tool_manager(
            blocked_tools=("memory", "cron"),
            include_subagent=False,
        )
        names = set(d.name for d in tm.get_definitions())
        assert "memory" not in names
        # cron 默认已由 include_cron=False 排除（不在注册列表），
        # 但仍加在 blocked_tools 确保安全
        assert "cron" not in names

    def test_always_blocks_cron(self):
        """子代理工具集始终不含 cron（include_cron=False）。"""
        tm = _build_child_tool_manager(
            blocked_tools=(),
            include_subagent=False,
        )
        names = set(d.name for d in tm.get_definitions())
        assert "cron" not in names

    def test_has_essential_tools(self):
        """子代理保留核心工具（文件操作、bash、search）。"""
        tm = _build_child_tool_manager(
            blocked_tools=("memory",),
            include_subagent=False,
        )
        names = set(d.name for d in tm.get_definitions())
        assert "read_file" in names
        assert "write_file" in names
        assert "bash" in names
        assert "search_files" in names


class TestAssembleChildLoop:
    """测试 assemble_child_loop 完整组装。"""

    def setup_method(self):
        self._saved = {
            "max_depth": Config.load().subagent_max_depth,
            "blocked": Config.load().subagent_blocked_tools,
        }

    def teardown_method(self):
        Config.load().subagent_max_depth = self._saved["max_depth"]
        # 恢复 blocked_tools 需要直接赋值 tuple
        Config.load().subagent_blocked_tools = self._saved["blocked"]

    def _make_loop(self, goal="test", context=None, depth=0, max_depth=1, blocked=None):
        parent_tm = __import__("kocor.tools.tool_manager", fromlist=["ToolManager"]).ToolManager()
        parent_tm.register_builtin_tools(include_cron=False)
        parent_tm.todo_store = None  # 模拟未设置

        parent_llm = MockLLM()
        return assemble_child_loop(
            goal=goal,
            context=context,
            parent_llm=parent_llm,
            parent_tool_manager=parent_tm,
            depth=depth,
            max_depth=max_depth,
            blocked_tools=blocked or ("memory",),
        )

    def test_loop_runs_goal_in_user_message(self):
        loop = self._make_loop(goal="测试目标")
        assert len(loop.ctx.messages) == 2
        assert loop.ctx.messages[0].role == "system"
        assert loop.ctx.messages[1].role == "user"
        assert "测试目标" in loop.ctx.messages[1].content

    def test_context_included_in_user_message(self):
        loop = self._make_loop(goal="修复 bug", context="文件路径: /tmp/a.py")
        content = loop.ctx.messages[1].content
        assert "文件路径: /tmp/a.py" in content
        assert "上下文:" in content

    def test_depth_zero_leaf_strips_subagent(self):
        Config.load().subagent_max_depth = 1
        loop = self._make_loop(goal="test", depth=0, max_depth=1)
        names = set(d.name for d in loop.ctx.tool_definitions)
        assert "subagent" not in names

    def test_depth_zero_orchestrator_keeps_subagent(self):
        Config.load().subagent_max_depth = 2
        loop = self._make_loop(goal="test", depth=0, max_depth=2)
        names = set(d.name for d in loop.ctx.tool_definitions)
        assert "subagent" in names

    def test_depth_one_orchestrator(self):
        Config.load().subagent_max_depth = 3
        # depth=1, max_depth=3 → 1+1=2 < 3 → orchestrator
        loop = self._make_loop(goal="test", depth=1, max_depth=3)
        names = set(d.name for d in loop.ctx.tool_definitions)
        assert "subagent" in names

    def test_depth_one_leaf_when_max_depth_2(self):
        Config.load().subagent_max_depth = 2
        # depth=1, max_depth=2 → 1+1=2 >= 2 → leaf
        loop = self._make_loop(goal="test", depth=1, max_depth=2)
        names = set(d.name for d in loop.ctx.tool_definitions)
        assert "subagent" not in names

    def test_system_prompt_is_focused(self):
        loop = self._make_loop(goal="搜索 bug")
        prompt = loop.ctx.messages[0].content
        assert "聚焦的子代理" in prompt
        assert "搜索 bug" in prompt
        assert "【输出要求】" in prompt

    def test_has_own_todo_store(self):
        loop = self._make_loop(goal="test")
        assert loop.tool_manager.todo_store is not None

    def test_noninteractive_permission(self):
        loop = self._make_loop(goal="test")
        assert loop.permission_mgr.policy == "noninteractive"

    def test_empty_hook_manager(self):
        loop = self._make_loop(goal="test")
        # HookManager 无注册钩子时，run 返回空列表
        assert len(loop.hook_manager._hooks) == 0

    def test_child_emitter_is_empty(self):
        loop = self._make_loop(goal="test")
        assert len(loop.event_emitter._subscribers) == 0

    def test_uses_parent_llm(self):
        parent_tm = __import__("kocor.tools.tool_manager", fromlist=["ToolManager"]).ToolManager()
        parent_tm.register_builtin_tools(include_cron=False)
        parent_llm = MockLLM()
        loop = assemble_child_loop(
            goal="test", context=None,
            parent_llm=parent_llm, parent_tool_manager=parent_tm,
            depth=0, max_depth=1,
        )
        assert loop.llm is parent_llm