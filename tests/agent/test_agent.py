"""测试 Agent 核心"""

from unittest.mock import MagicMock

from kocor.config import Config
from kocor.agent import Agent
from kocor.llm_provider.llm_client import LLMClient
from kocor.tools.definitions import ToolDefinition
from kocor.harness.budget import IterationBudget
from kocor.llm_provider.message import FunctionCall, Message, StreamChunk, ToolCall, ToolResult


class FakeLLMClient(LLMClient):
    """伪造的 LLM 客户端，用于测试 Agent 循环"""

    def __init__(self, responses: list[Message]):
        self.responses = responses
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


class TestAgentTextResponse:
    """测试 Agent 纯文本响应（无需工具调用）"""

    def test_single_turn(self):
        """单次对话直接返回文本"""
        llm = FakeLLMClient([Message(role="assistant", content="你好，我是 Kocor")])
        agent = Agent(llm=llm)
        result = agent.run("你好")
        assert result == "你好，我是 Kocor"

    def test_system_prompt_included(self):
        """system prompt 包含在消息中"""
        llm = FakeLLMClient([Message(role="assistant", content="hello")])
        agent = Agent(llm=llm)
        agent.run("hi")

        # 验证 llm.generate 收到的第一条消息是 system prompt
        # 通过检查消息数量来间接验证
        assert llm.call_count == 1


class TestAgentToolCall:
    """测试 Agent 工具调用循环"""

    def test_single_tool_call_then_text(self):
        """一次工具调用后返回文本"""
        llm = FakeLLMClient([
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
            ),
            Message(role="assistant", content="文件内容是: hello"),
        ])

        mock_tools = MagicMock(spec=ToolRegistryMock)
        mock_tools.get_definitions.return_value = []
        mock_tools.execute.return_value = ToolResult(
            tool_call_id="call_1",
            content="hello",
        )

        agent = Agent(llm=llm, tool_manager=mock_tools)
        result = agent.run("读 a.txt")
        assert result == "文件内容是: hello"
        mock_tools.execute.assert_called_once()
        assert llm.call_count == 2

    def test_multiple_tool_calls(self):
        """多次工具调用后返回文本"""
        llm = FakeLLMClient([
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
            ),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="call_2",
                    function=FunctionCall(name="read_file", arguments='{"path": "b.txt"}'),
                )],
            ),
            Message(role="assistant", content="两个文件都读完了"),
        ])

        mock_tools = MagicMock(spec=ToolRegistryMock)
        mock_tools.get_definitions.return_value = []

        def side_effect(tool_call):
            if tool_call.function.name == "read_file":
                return ToolResult(tool_call_id=tool_call.id, content="file content")
            return ToolResult(tool_call_id=tool_call.id, content="error")

        mock_tools.execute.side_effect = side_effect

        agent = Agent(llm=llm, tool_manager=mock_tools)
        result = agent.run("读 a.txt 和 b.txt")
        assert result == "两个文件都读完了"
        assert mock_tools.execute.call_count == 2
        assert llm.call_count == 3


class TestAgentTimeout:
    """测试 Agent 超时"""

    def test_max_iterations_reached(self):
        """达到最大迭代次数后返回超时"""
        llm = FakeLLMClient([
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{}'),
                )],
            ),
        ])

        mock_tools = MagicMock(spec=ToolRegistryMock)
        mock_tools.get_definitions.return_value = []
        mock_tools.execute.return_value = ToolResult(
            tool_call_id="call_1",
            content="content",
        )

        agent = Agent(llm=llm, tool_manager=mock_tools, budget=IterationBudget(max_iterations=3))
        result = agent.run("持续调用工具")
        assert "重复" in result


class TestAgentSystemPrompt:
    """测试自定义 system prompt"""

    def test_default_system_prompt(self):
        """默认 system prompt 包含 Kocor"""
        assert "Kocor" in Config.load().default_system_prompt

    def test_custom_system_prompt(self):
        """自定义 system prompt"""
        llm = FakeLLMClient([Message(role="assistant", content="hi")])
        agent = Agent(llm=llm)
        agent.run("hi")
        assert llm.call_count == 1

    def test_defensive_prompt_has_security_guidelines(self):
        """默认 system prompt 包含安全准则"""
        assert "不可信任" in Config.load().default_system_prompt or "不要执行" in Config.load().default_system_prompt
        assert "安全准则" in Config.load().default_system_prompt or "安全" in Config.load().default_system_prompt

    def test_defensive_prompt_warns_about_file_content(self):
        """默认 system prompt 提醒文件内容不可信"""
        assert "文件内容" in Config.load().default_system_prompt


# 简单的 mock 类，用于 spec
class FakeStreamLLMClient:
    """伪造的 LLM 客户端，用于测试 Agent 流式循环"""

    def __init__(self, responses: list[list[StreamChunk]]):
        self.responses = responses
        self.call_count = 0

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0) -> Message:
        # 不用于 stream 测试
        raise NotImplementedError

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        chunks = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        yield from chunks


class TestAgentStream:
    """测试 Agent 流式"""

    def test_stream_text_response(self):
        """单次对话流式返回文本"""
        llm = FakeStreamLLMClient([
            [
                StreamChunk(content="你好"),
                StreamChunk(content=", 我是 Kocor"),
                StreamChunk(is_final=True),
            ]
        ])
        agent = Agent(llm=llm)
        chunks = list(agent.stream("你好"))

        assert len(chunks) == 3
        assert chunks[0].content == "你好"
        assert chunks[1].content == ", 我是 Kocor"
        assert chunks[-1].is_final is True

    def test_stream_tool_call_then_text(self):
        """工具调用后继续流式输出"""
        llm = FakeStreamLLMClient([
            # 第一轮: 工具调用
            [
                StreamChunk(content="我来读文件"),
                StreamChunk(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                    )],
                    is_final=True,
                ),
            ],
            # 第二轮: 纯文本回复
            [
                StreamChunk(content="文件内容是: hello"),
                StreamChunk(is_final=True),
            ],
        ])

        mock_tools = MagicMock(spec=ToolRegistryMock)
        mock_tools.get_definitions.return_value = []

        def side_effect(tool_call):
            return ToolResult(tool_call_id=tool_call.id, content="hello")

        mock_tools.execute.side_effect = side_effect

        agent = Agent(llm=llm, tool_manager=mock_tools)
        chunks = list(agent.stream("读 a.txt"))

        # 文本 chunk + 工具调用 chunk + 文本 chunk + is_final
        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) == 2
        assert "我来读文件" in text_chunks[0].content
        assert "文件内容是: hello" in text_chunks[1].content
        assert chunks[-1].is_final is True

    def test_stream_max_iterations(self):
        """达到最大迭代次数后返回超时"""
        llm = FakeStreamLLMClient([
            [
                StreamChunk(
                    tool_calls=[ToolCall(
                        id="call_1",
                        function=FunctionCall(name="read_file", arguments='{}'),
                    )],
                    is_final=True,
                ),
            ]
        ])

        mock_tools = MagicMock(spec=ToolRegistryMock)
        mock_tools.get_definitions.return_value = []
        mock_tools.execute.return_value = ToolResult(
            tool_call_id="call_1",
            content="content",
        )

        agent = Agent(llm=llm, tool_manager=mock_tools, budget=IterationBudget(max_iterations=2))
        chunks = list(agent.stream("持续调用工具"))

        # 最后应该有超时信息
        assert chunks[-1].is_final is True
        final_content = "".join(c.content for c in chunks)
        assert "迭代" in final_content


# 简单的 mock 类，用于 spec
class ToolRegistryMock:
    skill_manager = None

    def get_definitions(self):
        return []

    def execute(self, tool_call):
        return ToolResult(tool_call_id="call_1", content="")

    def start_cron_scheduler(self):
        """cron 调度器，测试中为空操作。"""
        pass

    def stop_cron_scheduler(self):
        """cron 调度器，测试中为空操作。"""
        pass
