"""测试 Agent 与上下文管理集成。"""

from __future__ import annotations

from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message


class FakeLLMClient(LLMClient):
    """伪造的 LLM 客户端，用于测试 Agent 循环"""

    def __init__(self, responses: list[Message] | None = None):
        self.responses = responses or [Message(role="assistant", content="OK")]
        self.call_count = 0
        self.last_messages = None

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        self.last_messages = messages
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        raise NotImplementedError


class TestContextManagerIntegration:
    """测试 Agent 与 ContextBuilder 的集成。"""

    def test_default_agent_still_works(self):
        """默认配置的 Agent 应正常工作（向后兼容）。"""
        llm = FakeLLMClient()
        agent = Agent(llm=llm)
        result = agent.run("你好")
        assert result == "OK"

    def test_system_prompt_now_has_env_info(self):
        """使用 ContextBuilder 后 system prompt 应包含环境信息。"""
        llm = FakeLLMClient()
        agent = Agent(llm=llm)
        agent.run("你好")
        # 验证最后一次 LLM 请求的 system prompt
        system_msg = llm.last_messages[0]
        assert system_msg.role == "system"
        assert "当前工作目录" in system_msg.content

    def test_agent_with_memory_dir(self):
        """配置 memory_dir 后 Agent 应自动创建 MemoryManager。"""
        import tempfile
        from kocor.context.types import MemoryItem

        mem_dir = tempfile.mkdtemp()
        # 预先写入一条记忆
        from kocor.context.memory import MemoryManager
        memory = MemoryManager(memory_dir=mem_dir)
        memory.save(MemoryItem(
            name="user-info", description="用户信息", content="用户: 张三", memory_type="user",
        ))

        llm = FakeLLMClient()
        Config.set("memory_dir", mem_dir)
        try:
            agent = Agent(llm=llm)
            agent.run("你好")
        finally:
            Config.set("memory_dir", None)
        system_msg = llm.last_messages[0]
        assert "已记录的信息" in system_msg.content
        assert "用户: 张三" in system_msg.content

    def test_agent_context_strategy_does_not_break(self):
        """配置 context_strategy 不影响基本功能。"""
        llm = FakeLLMClient()
        agent = Agent(llm=llm)
        result = agent.run("测试")
        assert result == "OK"

    def test_agent_stream_with_context(self):
        """流式模式下 ContextBuilder 也应正常工作。"""
        class FakeStreamLLM:
            def __init__(self):
                self.call_count = 0
                self.last_messages = None

            @property
            def provider(self):
                return "fake"

            def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                raise NotImplementedError

            def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                self.last_messages = messages
                self.call_count += 1
                from kocor.llm_provider.message import StreamChunk
                yield StreamChunk(content="你好")
                yield StreamChunk(is_final=True)

        llm = FakeStreamLLM()
        agent = Agent(llm=llm)
        list(agent.stream("你好"))
        system_msg = llm.last_messages[0]
        assert system_msg.role == "system"
        assert "当前工作目录" in system_msg.content


class TestMultiTurnConversation:
    """测试多轮对话历史传递。"""

    def test_session_history_across_runs(self):
        """多次 run() 调用应传递会话历史。"""
        class HistoryTrackingLLM:
            def __init__(self):
                self.call_count = 0
                self.all_calls = []

            @property
            def provider(self):
                return "fake"

            def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                self.all_calls.append(list(messages))
                self.call_count += 1
                return Message(role="assistant", content=f"回答{self.call_count}")

            def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                raise NotImplementedError

        llm = HistoryTrackingLLM()
        agent = Agent(llm=llm)

        agent.run("第一轮问题")
        turn1_msgs = llm.all_calls[0]
        turn1_user = [m for m in turn1_msgs if m.role == "user"]
        assert any("第一轮问题" in m.content for m in turn1_user)

        agent.run("第二轮问题")
        turn2_msgs = llm.all_calls[1]
        turn2_users = [m for m in turn2_msgs if m.role == "user"]
        assert any("第一轮问题" in m.content for m in turn2_users)
        assert any("第二轮问题" in m.content for m in turn2_users)
        assert "assistant" in [m.role for m in turn2_msgs]

    def test_reset_conversation_clears_history(self):
        """reset_conversation() 后历史应清空。"""
        class TrackingLLM:
            def __init__(self):
                self.call_count = 0
                self.all_calls = []

            @property
            def provider(self):
                return "fake"

            def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                self.all_calls.append(list(messages))
                self.call_count += 1
                return Message(role="assistant", content=f"回答{self.call_count}")

            def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
                raise NotImplementedError

        llm = TrackingLLM()
        agent = Agent(llm=llm)

        agent.run("问题1")
        agent.reset_conversation()
        agent.run("问题2")

        turn2_msgs = llm.all_calls[1]
        turn2_users = [m for m in turn2_msgs if m.role == "user"]
        assert not any("问题1" in m.content for m in turn2_users)
        assert any("问题2" in m.content for m in turn2_users)