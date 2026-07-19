"""测试 Agent 的边缘情况 — 内置命令、slash 命令边界、_check_nudge、会话管理边界。

覆盖代码审查报告指出的「Agent 会话测试：覆盖 slash 命令 + 会话管理的组合场景」缺口。
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from tests.conftest import make_agent
from kocor.llm_provider.message import Message, StreamChunk, ToolResult
from kocor.skill.skill_manager import SkillManager
from kocor.skill.types import InvokeStrategy, SkillDefinition, SkillType
from kocor.tools.tool_manager import ToolManager

# ── Fake LLM ──


class FakeLLMClient(LLMClient):
    """伪造的 LLM 客户端。"""

    def __init__(self, responses: list[Message] | None = None):
        self.responses = responses or [Message(role="assistant", content="OK")]
        self.call_count = 0

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        resp = self.generate(messages, tools)
        yield StreamChunk(content=resp.content, is_final=True)


class ToolRegistryMock:
    skill_manager = None

    def get_definitions(self, filter_category=None):
        return []

    def execute(self, tool_call):
        return ToolResult(tool_call_id=tool_call.id, content="")


# ═══════════════════════════════════════════════
# 内置命令边界
# ═══════════════════════════════════════════════


class TestHandleBuiltinCommands:
    """_handle_builtin_commands 边界。"""

    def test_unknown_command_returns_none(self):
        """未知命令返回 None 走后续路由。"""
        agent = make_agent(llm=FakeLLMClient())
        result = agent._handle_builtin_commands("/unknown")
        assert result is None

    def test_reset_command(self):
        """reset 命令重置会话。"""
        llm = FakeLLMClient([Message(role="assistant", content="hi")])
        agent = make_agent(llm=llm)
        agent.run("你好")
        assert len(agent.context.session_history) > 0

        result = agent.run("/reset")
        assert "会话已重置" in result
        assert len(agent.context.session_history) == 0

    def test_new_command(self):
        """new 命令创建新会话。"""
        agent = make_agent(llm=FakeLLMClient())
        result = agent.run("/new")
        assert "新会话" in result

    def test_sessions_command_without_session_manager(self):
        """无 session_manager 时 /sessions 返回提示。"""
        agent = make_agent(llm=FakeLLMClient())
        result = agent.run("/sessions")
        # 没有 session_manager 时，/sessions 不是内置命令，走 skill 路由
        # 所以返回 None 对应内容
        assert result is not None

    def test_session_command_without_args(self):
        """无参数的 /session 命令。"""
        agent = make_agent(llm=FakeLLMClient())
        result = agent.run("/session")
        # 没有 session_manager 时，走 skill 路由
        assert result is not None

    def test_reset_after_new_resets(self):
        """连续 /reset 和 /new 都正常工作。"""
        agent = make_agent(llm=FakeLLMClient())
        r1 = agent.run("/reset")
        assert "会话已重置" in r1
        r2 = agent.run("/new")
        assert "新会话" in r2


# ═══════════════════════════════════════════════
# Slash 命令边界
# ═══════════════════════════════════════════════


class TestHandleSlashCommandEdgeCases:
    """_handle_slash_command 边界。"""

    def test_unknown_skill(self):
        """未知 skill 名称返回提示。"""
        skill_mgr = SkillManager()
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient(), tool_manager=tool_mgr)
        result = agent.run("/nonexistent")
        assert "Unknown skill" in result
        assert "nonexistent" in result

    def test_skill_with_wrong_invoke_strategy(self):
        """非 SLASH 策略的 skill 不可用 slash 调用。"""
        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="code_only",
            description="Only PROMPT skill",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.LLM,  # 非 SLASH
            handler=lambda: "hello",
        ))
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient(), tool_manager=tool_mgr)
        result = agent.run("/code_only")
        assert "cannot be invoked" in result

    def test_code_skill_returns_result(self):
        """CODE 类型 slash 技能返回执行结果。"""
        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="hello",
            description="Say hello",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
            handler=lambda user_input="": f"Hello, {user_input}!",
        ))
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient(), tool_manager=tool_mgr)
        result = agent.run("/hello world")
        assert result == "Hello, world!"

    def test_prompt_skill_runs_loop(self):
        """PROMPT 类型 slash 技能通过 Loop 执行。"""
        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="ask",
            description="Ask assistant",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.SLASH,
            prompt_template="回答：{user_input}",
            prompt_role="user",
        ))
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient([Message(role="assistant", content="这是回答")]), tool_manager=tool_mgr)
        result = agent.run("/ask 你好吗")
        assert result == "这是回答"

    def test_prompt_skill_system_role(self):
        """PROMPT 技能使用 system role 注入。"""
        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="system_prompt",
            description="System prompt skill",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.SLASH,
            prompt_template="你是专家",
            prompt_role="system",
        ))
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient([Message(role="assistant", content="专家回答")]), tool_manager=tool_mgr)
        result = agent.run("/system_prompt")
        assert result == "专家回答"

    def test_list_slash_skills(self):
        """_list_slash_skills 返回可用的 slash 技能列表。"""
        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="a_skill", description="A", skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH, handler=lambda: "a",
        ))
        skill_mgr.register(SkillDefinition(
            name="b_skill", description="B", skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.BOTH, handler=lambda: "b",
        ))
        # 非 SLASH 的不应列出
        skill_mgr.register(SkillDefinition(
            name="c_skill", description="C", skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.LLM, handler=lambda: "c",
        ))

        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient(), tool_manager=tool_mgr)
        result = agent._list_slash_skills()
        assert "/a_skill" in result
        assert "/b_skill" in result
        assert "/c_skill" not in result


# ═══════════════════════════════════════════════
# 流的 slash 命令
# ═══════════════════════════════════════════════


class TestStreamSlashCommand:
    """流模式下的 slash 命令处理。"""

    def test_stream_unknown_skill(self):
        """流模式中未知 skill 返回提示。"""
        skill_mgr = SkillManager()
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient(), tool_manager=tool_mgr)
        chunks = list(agent.stream("/unknown"))
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "Unknown skill" in chunks[0].content

    def test_stream_code_skill(self):
        """流模式中 CODE 技能返回结果。"""
        skill_mgr = SkillManager()
        skill_mgr.register(SkillDefinition(
            name="ping", description="Ping", skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH, handler=lambda: "pong",
        ))
        tool_mgr = ToolManager()
        tool_mgr.skill_manager = skill_mgr

        agent = make_agent(llm=FakeLLMClient(), tool_manager=tool_mgr)
        chunks = list(agent.stream("/ping"))
        assert len(chunks) == 1
        assert chunks[0].content == "pong"

    def test_stream_builtin_reset(self):
        """流模式中 /reset 内置命令。"""
        agent = make_agent(llm=FakeLLMClient())
        # 先运行一条消息建立会话
        agent.run("hello")
        assert len(agent.context.session_history) > 0

        chunks = list(agent.stream("/reset"))
        assert len(chunks) == 1
        assert "会话已重置" in chunks[0].content
        assert len(agent.context.session_history) == 0


# ═══════════════════════════════════════════════
# _check_nudge
# ═══════════════════════════════════════════════


class TestCheckNudge:
    """_check_nudge 记忆审查触发。"""

    def test_nudge_not_called_without_memory(self):
        """无 MemoryStore 时 nudge 不触发。"""
        agent = make_agent(llm=FakeLLMClient())
        agent._memory_store = None
        agent._background_reviewer = MagicMock()
        # 不会报错，也不会触发
        agent._check_nudge()
        assert agent._turns_since_memory == 0

    def test_nudge_interval_respected(self):
        """到达 nudge_interval 时触发审查。"""
        agent = make_agent(llm=FakeLLMClient())
        # 模拟 MemoryStore 和 BackgroundReviewer
        mock_reviewer = MagicMock()
        agent._background_reviewer = mock_reviewer
        agent._memory_store = MagicMock()  # _check_nudge 需要 _memory 非空
        agent._turns_since_memory = 0

        # 设置 nudge_interval 为 3
        original_interval = Config.load().nudge_interval
        Config.load().nudge_interval = 3

        try:
            # 第 1 次：_turns_since_memory=1
            agent._check_nudge()
            assert agent._turns_since_memory == 1
            mock_reviewer.review.assert_not_called()

            # 第 2 次：_turns_since_memory=2
            agent._check_nudge()
            assert agent._turns_since_memory == 2
            mock_reviewer.review.assert_not_called()

            # 第 3 次：达到间隔，触发
            agent._check_nudge()
            assert agent._turns_since_memory == 0
            mock_reviewer.review.assert_called_once()
        finally:
            Config.load().nudge_interval = original_interval

    def test_nudge_interval_default(self):
        """nudge_interval 默认值合理。"""
        assert Config.load().nudge_interval > 0


# ═══════════════════════════════════════════════
# 会话管理边界
# ═══════════════════════════════════════════════


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


@pytest.fixture
def session_manager(db_path):
    from kocor.session import SessionManager, SessionResetPolicy, SessionStore
    store = SessionStore(db_path=db_path)
    policy = SessionResetPolicy(mode="none")
    mgr = SessionManager(store=store, policy=policy)
    yield mgr
    if store.db:
        store.db.close()


class TestSessionManagementEdgeCases:
    """会话管理边界。"""

    def test_session_before_run_without_manager(self):
        """无 session_manager 时 _session_before_run 不报错。"""
        agent = make_agent(llm=FakeLLMClient())
        agent._session_before_run()  # 不应报错

    def test_session_after_run_without_manager(self):
        """无 session_manager 时 _session_after_run 不报错。"""
        agent = make_agent(llm=FakeLLMClient())
        agent._session_after_run()  # 不应报错

    def test_session_before_run_auto_reset_injects_notification(self, session_manager):
        """自动重置时注入通知消息。"""
        from datetime import datetime, timedelta

        from kocor.session import SessionManager, SessionResetPolicy

        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        sm = SessionManager(store=session_manager.store, policy=policy)

        agent = make_agent(llm=FakeLLMClient([Message(role="assistant", content="ok")]), session_manager=sm)

        # 第一次运行创建会话
        agent.run("first")

        # 模拟超时
        entry = sm.store.get_entry(sm.session_key)
        entry.updated_at = datetime.now() - timedelta(minutes=120)
        sm.store.set_entry(entry)

        # 第二次运行应触发自动重置
        agent.run("second")

        # 验证会话条目标记为自动重置
        info = sm.get_session_info(sm.session_key)
        assert info.was_auto_reset is True
        assert info.auto_reset_reason == "idle"

    def test_session_after_run_updates_metadata(self, session_manager):
        """_session_after_run 更新会话元数据。"""
        agent = make_agent(
            llm=FakeLLMClient([Message(role="assistant", content="ok")]),
            session_manager=session_manager,
        )
        agent.run("test")

        info = session_manager.get_session_info(session_manager.session_key)
        assert info is not None
        assert info.message_count > 0

    def test_session_after_run_handles_missing_entry(self, session_manager):
        """会话条目不存在时 _session_after_run 不报错。"""
        agent = make_agent(
            llm=FakeLLMClient([Message(role="assistant", content="ok")]),
            session_manager=session_manager,
        )

        # 直接调用 _session_after_run 而不先 run
        agent._session_after_run()  # 不应报错

    def test_persist_messages_after_multiple_runs(self, session_manager):
        """多次运行后消息被正确持久化。"""
        llm = FakeLLMClient([
            Message(role="assistant", content="回答 1"),
            Message(role="assistant", content="回答 2"),
            Message(role="assistant", content="回答 3"),
        ])
        agent = make_agent(llm=llm, session_manager=session_manager)

        agent.run("消息 1")
        agent.run("消息 2")
        agent.run("消息 3")

        info = session_manager.get_session_info(session_manager.session_key)
        messages = session_manager.load_messages(info.session_id)
        assert len(messages) >= 3

    def test_switch_session_with_invalid_id(self, session_manager):
        """切换到不存在的会话 ID 返回提示。"""
        agent = make_agent(
            llm=FakeLLMClient([Message(role="assistant", content="ok")]),
            session_manager=session_manager,
        )

        result = agent.run("/session nonexistent_id")
        assert "无可切换" in result or "未找到会话" in result or "无法切换" in result

    def test_format_sessions_table_empty(self):
        """空会话列表的格式化。"""
        table = Agent._format_sessions_table([])
        assert "暂无历史会话" in table

    def test_format_sessions_table_with_data(self):
        """有会话数据的格式化。"""
        sessions = [
            {"session_id": "sid_001", "created_at": "2026-07-10T10:00:00", "message_count": 3, "title": "测试会话"},
        ]
        table = Agent._format_sessions_table(sessions)
        assert "sid_001" in table
        assert "测试会话" in table
        assert "3" in table


# ═══════════════════════════════════════════════
# 停止/中断重置
# ═══════════════════════════════════════════════


class TestAgentStopReset:
    """Agent stop() 和 reset_conversation() 边界。"""

    def test_reset_conversation_clears_history(self):
        """reset_conversation() 清空历史。"""
        agent = make_agent(llm=FakeLLMClient([Message(role="assistant", content="ok")]))
        agent.run("hello")
        assert len(agent.context.session_history) > 0

        agent.reset_conversation()
        assert len(agent.context.session_history) == 0
        assert agent._persisted_msg_idx == 0

    def test_reset_conversation_with_session_manager(self, session_manager):
        """reset_conversation() 同时重置会话管理器。"""
        agent = make_agent(
            llm=FakeLLMClient([Message(role="assistant", content="ok")]),
            session_manager=session_manager,
        )
        agent.run("first")

        info_before = session_manager.get_session_info(session_manager.session_key)
        sid_before = info_before.session_id

        agent.reset_conversation()

        info_after = session_manager.get_session_info(session_manager.session_key)
        assert info_after.session_id != sid_before


# ═══════════════════════════════════════════════
# 轻量初始化和后向兼容
# ═══════════════════════════════════════════════


class TestAgentLightweightInit:
    """Agent 极简初始化。"""

    def test_llm_only(self):
        """仅传 llm 的极简初始化。"""
        agent = make_agent(llm=FakeLLMClient())
        assert agent.tool_manager is not None
        assert agent.tool_manager.permission_mgr is not None
        assert agent.hook_manager is not None
        assert agent.event_emitter is not None
        assert agent.max_iterations > 0
        assert agent.loop is not None

    def test_minimal_agent_runs(self):
        """极简初始化的 Agent 能运行。"""
        agent = make_agent(llm=FakeLLMClient([Message(role="assistant", content="hello")]))
        result = agent.run("hi")
        assert result == "hello"

    def test_minimal_agent_streams(self):
        """极简初始化的 Agent 能流式运行。"""
        llm = FakeLLMClient([Message(role="assistant", content="hello")])
        agent = make_agent(llm=llm)
        chunks = list(agent.stream("hi"))
        contents = [c.content for c in chunks if c.content]
        assert "hello" in "".join(contents)

    def test_agent_handle_builtin_commands_without_session(self):
        """无 session_manager 时内置命令不报错。"""
        agent = make_agent(llm=FakeLLMClient())

        # 不需要 session 的命令
        assert agent._handle_builtin_commands("/reset") is not None
        assert agent._handle_builtin_commands("/new") is not None

        # 需要 session 的命令返回 None（非内置，走后续路由）
        assert agent._handle_builtin_commands("/sessions") is None
        assert agent._handle_builtin_commands("/session test") is None