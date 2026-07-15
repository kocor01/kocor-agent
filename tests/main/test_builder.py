"""测试 AgentBuilder 装配逻辑。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


class TestAgentBuilderSubagent:
    """测试 Subagent 装配逻辑。"""

    def test_build_subagent_enabled(self, mock_llm_factory):
        """subagent_enabled=True 时 build_subagent 应设置 _subagent_runner。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        builder.build_llm()
        result = builder.build_subagent()
        assert result is builder
        assert builder.tool_manager._subagent_runner is not None

    def test_build_subagent_disabled(self, mock_llm_factory):
        """subagent_enabled=False 时 _subagent_runner 应为 None。"""
        cfg = Config.load()
        original = cfg.subagent_enabled
        cfg.subagent_enabled = False
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            builder.build_llm()
            builder.build_subagent()
            assert builder.tool_manager._subagent_runner is None
        finally:
            cfg.subagent_enabled = original


class TestAgentBuilderAssembly:
    """测试 AgentBuilder 组装逻辑。"""

    def test_build_llm(self, mock_llm_factory):
        """build_llm 应创建 LLM 并返回 self 以支持链式调用。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        result = builder.build_llm()
        assert result is builder
        assert builder.llm is not None

    def test_build_tools(self):
        """build_tools 应注册工具并返回 self。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        result = builder.build_tools()
        assert result is builder
        assert builder.tool_manager is not None

    def test_build_permission(self):
        """build_permission 应创建 PermissionManager 并返回 self。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        result = builder.build_permission()
        assert result is builder
        assert builder.permission_mgr is not None
        # 默认策略为 default
        assert builder.permission_mgr.policy == PermissionManager.POLICY_DEFAULT

    def test_build_hooks(self, mock_llm_factory):
        """build_hooks 应注册钩子并返回 self。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        logger = MagicMock()
        result = builder.build_llm().build_hooks(logger=logger)
        assert result is builder

    def test_build_session_enabled(self, monkeypatch):
        """session_enabled=True 时应有 session_manager。"""
        cfg = Config.load()
        original = cfg.session_enabled
        cfg.session_enabled = True
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            result = builder.build_session()
            assert result is builder
            assert builder.session_manager is not None
        finally:
            cfg.session_enabled = original

    def test_build_session_disabled(self, monkeypatch):
        """session_enabled=False 时 session_manager 应为 None。"""
        cfg = Config.load()
        original = cfg.session_enabled
        cfg.session_enabled = False
        try:
            from kocor._cli.builder import AgentBuilder
            builder = AgentBuilder()
            result = builder.build_session()
            assert result is builder
            assert builder.session_manager is None
        finally:
            cfg.session_enabled = original

    def test_full_assembly(self, mock_llm_factory):
        """完整链式调用应返回组装正确的 Agent。"""
        from kocor._cli.builder import AgentBuilder
        agent = (
            AgentBuilder()
            .build_llm()
            .build_tools()
            .build_permission()
            .build()
        )
        assert isinstance(agent, Agent)
        assert agent.llm is not None
        assert agent.tool_manager is not None
        assert agent.permission_mgr is not None

    def test_agent_can_run(self, mock_llm_factory):
        """组装后的 Agent 应能正常执行 run()。"""
        from kocor._cli.builder import AgentBuilder
        agent = (
            AgentBuilder()
            .build_llm()
            .build_tools()
            .build_permission()
            .build()
        )
        result = agent.run("hello")
        assert result == "ok"

    def test_builder_chain_returns_self(self, mock_llm_factory):
        """链式调用各方法应返回 self。"""
        from kocor._cli.builder import AgentBuilder
        builder = AgentBuilder()
        assert builder.build_llm() is builder
        assert builder.build_tools() is builder
        assert builder.build_permission() is builder
        logger = MagicMock()
        assert builder.build_hooks(logger=logger) is builder
        assert builder.build_session() is builder
        assert builder.build() is not builder  # build() 返回 Agent，不是 self