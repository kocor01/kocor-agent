"""测试 AgentBuilder 装配逻辑。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition
from kocor.tools.permission import PermissionManager


class FakeLLMClient(LLMClient):
    """伪造的 LLM 客户端，用于测试 AgentBuilder。"""

    def __init__(self, responses: list[Message] | None = None):
        self.responses = responses or [Message(role="assistant", content="ok")]
        self.call_count = 0

    @property
    def provider(self) -> str:
        return "fake"

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Message:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp


@pytest.fixture
def mock_llm_factory(monkeypatch):
    """Mock LlmFactory.create() 返回 FakeLLMClient。"""
    fake = FakeLLMClient()
    from kocor.llm_provider.llm_factory import LlmFactory
    monkeypatch.setattr(LlmFactory, "create", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _skip_mcp_skill(monkeypatch):
    """Builder 单元测试无需 MCP 服务器连接或技能加载（ToolManager.register_all 默认会启动 MCP）。

    各 build_* 方法只验证装配逻辑，不验证 MCP/Skill 功能。
    """
    from kocor.tools import tool_manager as tm_mod

    def _fast_register_all(self, include_subagent=False):
        # 仅注册内置工具，跳过耗时的 MCP 服务器和技能加载
        self.register_builtin_tools(include_subagent=include_subagent)

    monkeypatch.setattr(tm_mod.ToolManager, "register_all", _fast_register_all)


class TestAgentBuilderSubagent:
    """测试 Subagent 装配逻辑。"""

    def test_init_subagent_enabled(self, mock_llm_factory):
        """subagent_enabled=True 时 _init_subagent 应设置 _subagent_runner。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder._init_llm()
        builder._init_subagent()
        assert builder.tool_manager._subagent_runner is not None

    def test_init_subagent_disabled(self, mock_llm_factory):
        """subagent_enabled=False 时不应创建 SubagentRunner。"""
        cfg = Config.load()
        original = cfg.subagent_enabled
        cfg.subagent_enabled = False
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            builder._init_llm()
            builder._init_subagent()
            # tool_manager 未被创建，_subagent_runner 不存在
            assert builder.tool_manager is None
        finally:
            cfg.subagent_enabled = original


class TestAgentBuilderAssembly:
    """测试 AgentBuilder 组装逻辑。"""

    def test_init_llm(self, mock_llm_factory):
        """_init_llm 应创建 LLM。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder._init_llm()
        assert builder.llm is not None

    def test_init_tool_manager(self):
        """_init_tool_manager 应注册工具。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder._init_tool_manager()
        assert builder.tool_manager is not None

    def test_init_permission_mgr(self):
        """_init_permission_mgr 应创建 PermissionManager。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder._init_permission_mgr()
        assert builder.permission_mgr is not None
        # 默认策略为 default
        assert builder.permission_mgr.policy == PermissionManager.POLICY_DEFAULT

    def test_init_hook_manager(self, mock_llm_factory):
        """_init_hook_manager 应注册钩子。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        logger = MagicMock()
        builder._init_llm()
        builder._init_hook_manager(logger=logger)

    def test_init_session_manager_enabled(self, monkeypatch):
        """session_enabled=True 时应有 session_manager。"""
        cfg = Config.load()
        original = cfg.session_enabled
        cfg.session_enabled = True
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            builder._init_session_manager()
            assert builder.session_manager is not None
        finally:
            cfg.session_enabled = original

    def test_init_session_manager_disabled(self, monkeypatch):
        """session_enabled=False 时 session_manager 应为 None。"""
        cfg = Config.load()
        original = cfg.session_enabled
        cfg.session_enabled = False
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            builder._init_session_manager()
            assert builder.session_manager is None
        finally:
            cfg.session_enabled = original

    def test_full_assembly(self, mock_llm_factory):
        """完整 build 流程应返回组装正确的 Agent。"""
        from kocor._cli.builder import AgentBuilder
        logger = MagicMock()
        agent = AgentBuilder().build(logger=logger)
        assert isinstance(agent, Agent)
        assert agent.llm is not None
        assert agent.tool_manager is not None
        assert agent.permission_mgr is not None
        # 新增组件：ctx、todo_store 应已装配
        assert agent.context is not None
        assert agent._todo_store is not None
        assert agent.tool_manager.todo_store is agent._todo_store

    def test_agent_can_run(self, mock_llm_factory):
        """组装后的 Agent 应能正常执行 run()。"""
        from kocor._cli.builder import AgentBuilder
        logger = MagicMock()
        agent = AgentBuilder().build(logger=logger)
        result = agent.run("hello")
        assert result == "ok"

    def test_init_returns_agent(self, mock_llm_factory):
        """build() 应返回 Agent 实例。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        result = builder.build(logger=MagicMock())
        assert isinstance(result, Agent)


class TestAgentBuilderMemoryTodo:
    """测试 Memory、Todo、Context 装配逻辑。"""

    def test_init_todo_store(self, mock_llm_factory):
        """_init_todo_store 应创建 TodoStore 并注入 tool_manager。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder._init_todo_store()
        assert builder._todo_store is not None
        assert builder.tool_manager.todo_store is builder._todo_store

    def test_init_todo_store_wired_to_context(self, mock_llm_factory):
        """完整 build 后 todo_store 应注入到 ctx。"""
        from kocor._cli.builder import AgentBuilder
        logger = MagicMock()
        agent = AgentBuilder().build(logger=logger)
        assert agent.context.todo_store is agent._todo_store

    def test_init_memory_disabled(self, mock_llm_factory):
        """memory_enabled=False 时 _init_memory 不创建 MemoryStore。"""
        cfg = Config.load()
        original = cfg.memory_enabled
        cfg.memory_enabled = False
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            builder._init_llm()
            builder._init_memory()
            assert builder._memory is None
            assert builder._background_reviewer is None
        finally:
            cfg.memory_enabled = original

    def test_init_context(self, mock_llm_factory):
        """_init_context 应创建 ContextManager。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder._init_llm()
        builder._init_todo_store()
        builder._init_context()
        assert builder.context is not None
        assert builder.context.todo_store is builder._todo_store