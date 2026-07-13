"""测试 cron worker 子进程内的独立 Agent 装配（agent_builder）。
"""

from __future__ import annotations

import pytest

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class FakeLLMClient(LLMClient):
    """伪造的 LLM 客户端，用于测试 agent_builder。"""

    def __init__(self, responses: list[Message] | None = None):
        self.responses = responses or [Message(role="assistant", content="cron-ok")]
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


@pytest.fixture
def disable_memory():
    """禁用 memory_enabled（子进程不应加载人类记忆）。"""
    cfg = Config.load()
    orig = cfg.memory_enabled
    cfg.memory_enabled = False
    yield
    cfg.memory_enabled = orig


class TestBuildCronAgent:
    """测试 build_cron_agent 装配逻辑。"""

    def test_omits_cronjob_tool(self, mock_llm_factory, disable_memory):
        """cron worker 的 ToolManager 不应注册 cronjob 工具（防递归）。"""
        from kocor.tools.toolsets.cron.agent_builder import build_cron_agent

        agent, scheduler = build_cron_agent()
        names = {d.name for d in agent.tool_manager.get_definitions()}
        assert "cronjob" not in names

    def test_returns_agent_and_scheduler(self, mock_llm_factory, disable_memory):
        """返回 (Agent, CronScheduler) 元组。"""
        from kocor.agent import Agent
        from kocor.tools.toolsets.cron.agent_builder import build_cron_agent
        from kocor.tools.toolsets.cron.scheduler import CronScheduler

        agent, scheduler = build_cron_agent()
        assert isinstance(agent, Agent)
        assert isinstance(scheduler, CronScheduler)

    def test_scheduler_holds_agent_ref(self, mock_llm_factory, disable_memory):
        """CronScheduler.agent 即 Agent 实例。"""
        from kocor.tools.toolsets.cron.agent_builder import build_cron_agent

        agent, scheduler = build_cron_agent()
        assert scheduler.agent is agent

    def test_isolated_instances(self, mock_llm_factory, disable_memory):
        """两次 build 产生独立的 ToolManager（配置隔离）。"""
        from kocor.tools.toolsets.cron.agent_builder import build_cron_agent

        a1, _ = build_cron_agent()
        a2, _ = build_cron_agent()
        assert a1.tool_manager is not a2.tool_manager