"""测试 Agent 会话集成。"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message, ToolResult
from kocor.session import SessionManager, SessionResetPolicy, SessionStore
from kocor.tools.definitions import ToolDefinition
from tests.agent.test_agent import FakeLLMClient


@pytest.fixture
def db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


@pytest.fixture
def session_manager(db_path) -> SessionManager:
    store = SessionStore(db_path=db_path)
    policy = SessionResetPolicy(mode="none")
    mgr = SessionManager(store=store, policy=policy)
    yield mgr
    if store.db:
        store.db.close()


class TestAgentSessionIntegration:
    """Agent 与 SessionManager 集成测试。"""

    def test_agent_with_session_manager(self, session_manager):
        """带 SessionManager 的 Agent 正常运行。"""
        llm = FakeLLMClient([Message(role="assistant", content="你好，我是 Kocor")])
        agent = Agent(llm=llm, session_manager=session_manager)
        result = agent.run("你好")
        assert result == "你好，我是 Kocor"

    def test_agent_update_session_after_run(self, session_manager):
        """运行后会话元数据应更新。"""
        llm = FakeLLMClient([Message(role="assistant", content="OK")])
        agent = Agent(llm=llm, session_manager=session_manager)
        agent.run("test")

        info = session_manager.get_session_info(session_manager.session_key)
        assert info is not None
        assert info.message_count > 0
        assert info.total_tokens >= 0

    def test_agent_reset_via_command(self, session_manager):
        """Agent 接收 /reset 应重置会话。"""
        llm = FakeLLMClient([Message(role="assistant", content="OK")])
        agent = Agent(llm=llm, session_manager=session_manager)
        agent.run("第一条消息")

        info_before = session_manager.get_session_info(session_manager.session_key)
        assert info_before is not None
        session_id_before = info_before.session_id

        result = agent.run("/reset")
        assert result == "✅ 会话已重置。"

        info_after = session_manager.get_session_info(session_manager.session_key)
        assert info_after is not None
        # 会话应被重置，新的 session_id
        assert info_after.session_id != session_id_before
        assert info_after.is_fresh_reset is True

    def test_agent_reset_conversation_method(self, session_manager):
        """reset_conversation() 应同时重置会话和上下文。"""
        llm = FakeLLMClient([Message(role="assistant", content="OK")])
        agent = Agent(llm=llm, session_manager=session_manager)
        agent.run("消息")
        assert len(agent.ctx.session_history) > 0

        info_before = session_manager.get_session_info(session_manager.session_key)
        sid_before = info_before.session_id

        agent.reset_conversation()

        assert len(agent.ctx.session_history) == 0

        info_after = session_manager.get_session_info(session_manager.session_key)
        assert info_after.session_id != sid_before

    def test_agent_message_persisted(self, session_manager):
        """运行后消息应持久化到 SQLite。"""
        llm = FakeLLMClient([Message(role="assistant", content="I am Kocor")])
        agent = Agent(llm=llm, session_manager=session_manager)
        agent.run("你是谁？")

        info = session_manager.get_session_info(session_manager.session_key)
        assert info is not None

        messages = session_manager.load_messages(info.session_id)
        assert len(messages) >= 1
        assert messages[0].role == "user"
        assert messages[0].content == "你是谁？"

    def test_agent_sessions_builtin_command(self, session_manager):
        """/sessions 命令应返回列表（含"暂无"提示）。"""
        llm = FakeLLMClient([Message(role="assistant", content="hello")])
        agent = Agent(llm=llm, session_manager=session_manager)

        # 无历史时
        result = agent.run("/sessions")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_agent_session_switch_builtin_command(self, session_manager):
        """/session 命令应能切换会话。"""
        llm = FakeLLMClient([Message(role="assistant", content="hello")])
        agent = Agent(llm=llm, session_manager=session_manager)

        # 创建第一个会话
        agent.run("消息 A")
        info1 = session_manager.get_session_info(session_manager.session_key)
        sid1 = info1.session_id

        # 重置创建第二个会话
        agent.run("/reset")
        info2 = session_manager.get_session_info(session_manager.session_key)
        sid2 = info2.session_id
        assert sid2 != sid1

        # 切换到会话 1
        result = agent.run(f"/session {sid1}")
        assert "已切换到" in result
        assert sid1 in result

        # 验证当前会话指向 sid1
        info_after = session_manager.get_session_info(session_manager.session_key)
        assert info_after.session_id == sid1

    def test_agent_multiple_runs_same_session(self, session_manager):
        """多次 run() 应在同一会话中累积消息。"""
        llm = FakeLLMClient([
            Message(role="assistant", content="回复 1"),
            Message(role="assistant", content="回复 2"),
            Message(role="assistant", content="回复 3"),
        ])
        agent = Agent(llm=llm, session_manager=session_manager)
        agent.run("第 1 条")
        agent.run("第 2 条")
        agent.run("第 3 条")

        info = session_manager.get_session_info(session_manager.session_key)
        assert info is not None
        assert info.message_count >= 3

        messages = session_manager.load_messages(info.session_id)
        assert len(messages) >= 3

    def test_agent_without_session_manager_still_works(self):
        """不传 session_manager 时原有行为不变。"""
        llm = FakeLLMClient([Message(role="assistant", content="OK")])
        agent = Agent(llm=llm)
        result = agent.run("hello")
        assert result == "OK"

    def test_agent_reset_without_session_manager_still_works(self):
        """不传 session_manager 时 /reset 走原有流程。"""
        llm = FakeLLMClient([Message(role="assistant", content="OK")])
        agent = Agent(llm=llm)

        # 没有 session_manager 时 /reset 应返回"Unknown"（因为没有注册 skill）
        result = agent.run("/reset")
        # 实际上，/reset 现在由内置命令处理，即使没有 session_manager
        # 它会调用 reset_conversation() 并返回提示
        assert "会话已重置" in result or "Unknown" in result or "reset" in result.lower()

    def test_agent_session_auto_reset_notification(self, db_path):
        """自动重置时应在上下文中注入通知消息。"""
        store = SessionStore(db_path=db_path)
        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        sm = SessionManager(store=store, policy=policy, profile="test-auto-reset")

        llm = FakeLLMClient([Message(role="assistant", content="OK")])
        agent = Agent(llm=llm, session_manager=sm)

        # 创建会话
        agent.run("测试")

        # 手动修改 updated_at 来模拟超时
        entry = sm.store.get_entry(sm.session_key)
        from datetime import timedelta
        entry.updated_at = datetime.now() - timedelta(minutes=120)
        sm.store.set_entry(entry)

        # 再次运行应触发自动重置
        agent.run("新消息")

        info = sm.get_session_info(sm.session_key)
        assert info.was_auto_reset is True
        assert info.auto_reset_reason == "idle"
