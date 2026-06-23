"""测试 Agent 与上下文管理集成。"""

from __future__ import annotations

import os
import tempfile

from kocor.agent import Agent
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.tool_definition import ToolDefinition
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


class TestAgentContextIntegration:
    """测试 Agent 与 ContextBuilder 的集成。"""

    def test_default_agent_still_works(self):
        """默认配置的 Agent 应正常工作（向后兼容）。"""
        llm = FakeLLMClient()
        agent = Agent(llm=llm, max_iterations=20)
        result = agent.run("你好")
        assert result == "OK"

    def test_system_prompt_now_has_env_info(self):
        """使用 ContextBuilder 后 system prompt 应包含环境信息。"""
        llm = FakeLLMClient()
        agent = Agent(llm=llm, max_iterations=20)
        agent.run("你好")
        # 验证最后一次 LLM 请求的 system prompt
        system_msg = llm.last_messages[0]
        assert system_msg.role == "system"
        assert "当前工作目录" in system_msg.content

    def test_agent_with_memory_dir(self):
        """配置 memory_dir 后 Agent 应自动创建 MemoryManager。"""
        import tempfile
        from kocor.context.models import MemoryItem

        mem_dir = tempfile.mkdtemp()
        # 预先写入一条记忆
        from kocor.context.memory import MemoryManager
        memory = MemoryManager(memory_dir=mem_dir)
        memory.save(MemoryItem(
            name="user-info", description="用户信息", content="用户: 张三", memory_type="user",
        ))

        llm = FakeLLMClient()
        agent = Agent(
            llm=llm,
            max_iterations=20,
            memory_dir=mem_dir,
        )
        agent.run("你好")
        system_msg = llm.last_messages[0]
        assert "已记录的信息" in system_msg.content
        assert "用户: 张三" in system_msg.content

    def test_agent_with_custom_project_instructions(self):
        """自定义项目指令文件应被加载到 system prompt。"""
        content = "项目规范：使用 Python 3.12"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(content)
            path = f.name

        try:
            llm = FakeLLMClient()
            agent = Agent(
                llm=llm,
                max_iterations=20,
                project_instructions_path=path,
            )
            agent.run("你好")
            system_msg = llm.last_messages[0]
            assert "项目指令" in system_msg.content
            assert "Python 3.12" in system_msg.content
        finally:
            os.unlink(path)

    def test_agent_context_strategy_does_not_break(self):
        """配置 context_strategy 不影响基本功能。"""
        llm = FakeLLMClient()
        agent = Agent(
            llm=llm,
            max_iterations=20,
            context_strategy="default",
        )
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
        agent = Agent(llm=llm, max_iterations=20)
        list(agent.stream("你好"))
        system_msg = llm.last_messages[0]
        assert system_msg.role == "system"
        assert "当前工作目录" in system_msg.content