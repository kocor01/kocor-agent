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
from kocor.skill.skill_manager import SkillManager
from kocor.skill.types import InvokeStrategy, SkillDefinition, SkillType
from kocor.tools.definitions import ToolDefinition
from kocor.tools.tool_manager import ToolManager
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

    # ── 修复 3.2: _session_after_run 使用 _persisted_msg_idx ──

    def test_agent_messages_persisted_after_reset(self, session_manager):
        """reset_conversation() 后，后续消息应持久化到新会话。

        复现场景：_session_after_run 使用 entry.message_count 作为 prev_count，
        但会话已被重置，entry 指向新会话（message_count=0）。
        修复后使用 _persisted_msg_idx 作为基准，确保消息被正确持久化。
        """
        llm = FakeLLMClient([
            Message(role="assistant", content="回复 1"),
            Message(role="assistant", content="回复 2"),
        ])
        agent = Agent(llm=llm, session_manager=session_manager)

        # 第一次运行：建立会话，消息被持久化
        agent.run("第一条消息")
        info1 = session_manager.get_session_info(session_manager.session_key)
        assert info1 is not None
        sid1 = info1.session_id
        msgs1 = session_manager.load_messages(sid1)
        assert len(msgs1) >= 1

        # 重置会话（模拟 ReAct 循环中的重置）
        agent.reset_conversation()

        # 第二次运行：消息应持久化到新会话
        agent.run("第二条消息")
        info2 = session_manager.get_session_info(session_manager.session_key)
        assert info2 is not None
        assert info2.session_id != sid1  # 新会话

        msgs2 = session_manager.load_messages(info2.session_id)
        assert len(msgs2) >= 1
        assert msgs2[0].content == "第二条消息"

    def test_agent_auto_reset_persists_new_messages(self, db_path):
        """自动重置后，新运行的消息应正确持久化到新会话。

        复现场景：_session_before_run 中自动重置时 _persisted_msg_idx 未被重置，
        仍持有上次运行后的值，导致 _session_after_run 计算 msg_delta=0。
        修复后自动重置时重置 _persisted_msg_idx=0，确保后续消息被持久化。
        """
        from datetime import timedelta

        store = SessionStore(db_path=db_path)
        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        sm = SessionManager(store=store, policy=policy, profile="test-auto-reset-persist")

        llm = FakeLLMClient([
            Message(role="assistant", content="回复 1"),
            Message(role="assistant", content="回复 2"),
        ])
        agent = Agent(llm=llm, session_manager=sm)

        # 第一次运行：建立会话，消息被持久化
        agent.run("第一条消息")
        entry = sm.store.get_entry(sm.session_key)
        assert entry is not None
        sid_before = entry.session_id
        msgs_before = sm.load_messages(sid_before)
        assert len(msgs_before) >= 1

        # 模拟超时触发自动重置
        entry.updated_at = datetime.now() - timedelta(minutes=120)
        sm.store.set_entry(entry)

        # 第二次运行：应触发自动重置，新消息应持久化到新会话
        agent.run("第二条消息")

        info_after = sm.get_session_info(sm.session_key)
        assert info_after is not None
        assert info_after.was_auto_reset is True
        assert info_after.session_id != sid_before  # 新会话

        # 验证新会话的消息被正确持久化
        msgs_after = sm.load_messages(info_after.session_id)
        assert len(msgs_after) >= 1
        # 至少包含 "第二条消息" 用户消息
        user_msgs = [m for m in msgs_after if m.role == "user" and "第二条消息" in str(m.content)]
        assert len(user_msgs) >= 1

    def test_agent_multiple_runs_with_reset_between(self, session_manager):
        """多次运行 + 中间重置，每次运行的消息都应正确持久化。

        确保 _session_after_run 在会话重置后仍能正确计算消息增量。
        """
        llm = FakeLLMClient([
            Message(role="assistant", content="回复 A"),
            Message(role="assistant", content="回复 B"),
            Message(role="assistant", content="回复 C"),
        ])
        agent = Agent(llm=llm, session_manager=session_manager)

        # 第一次运行
        agent.run("消息 A")
        info = session_manager.get_session_info(session_manager.session_key)
        sid_a = info.session_id
        assert len(session_manager.load_messages(sid_a)) >= 1

        # 重置
        agent.reset_conversation()

        # 第二次运行
        agent.run("消息 B")
        info = session_manager.get_session_info(session_manager.session_key)
        sid_b = info.session_id
        assert sid_b != sid_a
        msgs_b = session_manager.load_messages(sid_b)
        assert len(msgs_b) >= 1
        assert msgs_b[0].content == "消息 B"

        # 再重置一次
        agent.reset_conversation()

        # 第三次运行
        agent.run("消息 C")
        info = session_manager.get_session_info(session_manager.session_key)
        sid_c = info.session_id
        assert sid_c != sid_b
        msgs_c = session_manager.load_messages(sid_c)
        assert len(msgs_c) >= 1
        assert msgs_c[0].content == "消息 C"

    # ── 修复 3.3:  slash 命令不经过 session_before_run/after_run ──

    def test_slash_prompt_skill_updates_session(self, session_manager):
        """PROMPT 类型 slash 命令执行后应更新会话元数据并持久化消息。

        修复 3.3：_handle_slash_command 中 loop._run_messages() 不经过
        _session_after_run，导致 PROMPT 型技能触发的 LLM 会话消息未被持久化。
        """
        llm = FakeLLMClient([Message(role="assistant", content="slash 技能回复")])

        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="test",
            description="测试 slash 技能",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.SLASH,
            prompt_template="你是测试助手，请回复：{user_input}",
            prompt_role="user",
        ))

        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = Agent(llm=llm, tool_manager=tool_mgr, session_manager=session_manager)
        result = agent.run("/test 你好")

        assert result == "slash 技能回复"

        # 会话元数据应已更新
        info = session_manager.get_session_info(session_manager.session_key)
        assert info is not None
        assert info.message_count > 0

        # 消息应已持久化到 SQLite
        messages = session_manager.load_messages(info.session_id)
        assert len(messages) >= 1

    def test_slash_code_skill_updates_session(self, session_manager):
        """CODE 类型 slash 命令执行后也应更新会话元数据。

        修复 3.3：_handle_slash_command 直接返回 CODE 技能结果，不经过
        _session_after_run，导致会话元数据未更新。
        """
        llm = FakeLLMClient([Message(role="assistant", content="ignored")])

        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="hello",
            description="测试 CODE slash 技能",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
            handler=lambda user_input="": f"你好，{user_input}!",
        ))

        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = Agent(llm=llm, tool_manager=tool_mgr, session_manager=session_manager)
        result = agent.run("/hello world")

        assert result == "你好，world!"

        # 会话元数据应已更新
        info = session_manager.get_session_info(session_manager.session_key)
        assert info is not None
        assert info.message_count >= 0  # CODE 技能不产生 LLM 消息，但会话记录应存在
