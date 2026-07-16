"""测试 BackgroundReviewer。"""

from __future__ import annotations

from kocor.llm_provider.message import Message, ToolCall
from kocor.memory.reviewer import MEMORY_REVIEW_PROMPT, BackgroundReviewer
from kocor.memory.store import MemoryStore


class _FakeReviewLLM:
    """模拟 LLM 用于测试。"""

    def __init__(self, should_memorize: bool = False):
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.should_memorize = should_memorize

    @property
    def provider(self):
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        if self.should_memorize:
            return Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=type(
                            "Func",
                            (),
                            {
                                "name": "memory",
                                "arguments": (
                                '{"operations":[{"action":"add",'
                                '"target":"user","content":"User named Alice"}]}'
                            ),
                            },
                        )(),
                    ),
                ],
            )
        return Message(role="assistant", content="Nothing to save.")

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        raise NotImplementedError


class TestBackgroundReviewer:
    """测试后台审查器。"""

    def test_review_nothing_to_save(self, tmp_path):
        """无值得记忆的对话不应写入。"""
        store = MemoryStore(memory_dir=str(tmp_path), memory_limit=2200, user_limit=1375, user_enabled=True)
        store.load_from_disk()
        llm = _FakeReviewLLM(should_memorize=False)
        reviewer = BackgroundReviewer(llm=llm, store=store)

        reviewer.review(
            [
                Message(role="user", content="你好"),
                Message(role="assistant", content="你好，有什么可以帮助你的？"),
            ]
        )

        assert llm.call_count == 1
        assert store.list_entries("memory") == []  # 无写入

    def test_review_saves_when_llm_decides(self, tmp_path):
        """审查发现值得记忆的内容应写入存储。"""
        store = MemoryStore(memory_dir=str(tmp_path), memory_limit=2200, user_limit=1375, user_enabled=True)
        store.load_from_disk()
        llm = _FakeReviewLLM(should_memorize=True)
        reviewer = BackgroundReviewer(llm=llm, store=store)

        reviewer.review(
            [
                Message(role="user", content="我叫张三"),
                Message(role="assistant", content="你好张三！"),
            ]
        )

        assert llm.call_count == 1
        entries = store.list_entries("user")
        assert len(entries) == 1
        assert "Alice" in entries[0]

    def test_review_includes_memory_tool_in_tools(self, tmp_path):
        """审查时 LLM 应能看到 memory 工具。"""
        store = MemoryStore(memory_dir=str(tmp_path), memory_limit=2200, user_limit=1375, user_enabled=True)
        store.load_from_disk()
        llm = _FakeReviewLLM(should_memorize=False)
        reviewer = BackgroundReviewer(llm=llm, store=store)

        reviewer.review([Message(role="user", content="hi")])

        assert llm.last_tools is not None
        assert any(t.name == "memory" for t in llm.last_tools)

    def test_review_prompt_in_system(self, tmp_path):
        """审查的 system prompt 应包含审查提示。"""
        store = MemoryStore(memory_dir=str(tmp_path), memory_limit=2200, user_limit=1375, user_enabled=True)
        store.load_from_disk()
        llm = _FakeReviewLLM(should_memorize=False)
        reviewer = BackgroundReviewer(llm=llm, store=store)

        reviewer.review([Message(role="user", content="hi")])

        system_msgs = [m for m in llm.last_messages if m.role == "system"]
        assert any(MEMORY_REVIEW_PROMPT in m.content for m in system_msgs)

    def test_agent_with_nudge(self, tmp_path):
        """Agent 在 N 轮后应触发后台审查。"""
        from kocor.agent import Agent
        from kocor.config import Config
        from tests.agent.test_agent_context import FakeLLMClient

        store = MemoryStore(memory_dir=str(tmp_path), memory_limit=2200, user_limit=1375, user_enabled=True)
        store.load_from_disk()

        llm = FakeLLMClient()
        Config.load().memory_dir = str(tmp_path)
        try:
            agent = Agent(llm=llm)
            # 注入 counter 为 nudge_interval - 1
            agent._turns_since_memory = 1
            agent.run("你好")
            # nudge_interval=10 时不应触发
        finally:
            Config.load().memory_dir = None
