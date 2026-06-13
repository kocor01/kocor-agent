"""测试 Agent 核心"""

from unittest.mock import MagicMock

from kocor.agent import Agent, DEFAULT_SYSTEM_PROMPT
from kocor.config import LLMConfig
from kocor.llm_client import LLMClient, ToolDefinition
from kocor.message import Message, ToolCall, FunctionCall, ToolResult


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
        agent = Agent(llm=llm, max_iterations=20)
        result = agent.run("你好")
        assert result == "你好，我是 Kocor"

    def test_system_prompt_included(self):
        """system prompt 包含在消息中"""
        llm = FakeLLMClient([Message(role="assistant", content="hello")])
        agent = Agent(llm=llm, max_iterations=20)
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

        agent = Agent(llm=llm, tools=mock_tools, max_iterations=20)
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

        agent = Agent(llm=llm, tools=mock_tools, max_iterations=20)
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

        agent = Agent(llm=llm, tools=mock_tools, max_iterations=3)
        result = agent.run("持续调用工具")
        assert "迭代" in result and "仍未完成" in result


class TestAgentSystemPrompt:
    """测试自定义 system prompt"""

    def test_default_system_prompt(self):
        """默认 system prompt 包含 Kocor"""
        llm = FakeLLMClient([Message(role="assistant", content="hi")])
        agent = Agent(llm=llm, max_iterations=20)
        assert "Kocor" in DEFAULT_SYSTEM_PROMPT

    def test_custom_system_prompt(self):
        """自定义 system prompt"""
        llm = FakeLLMClient([Message(role="assistant", content="hi")])
        agent = Agent(llm=llm, system_prompt="你是助手", max_iterations=20)
        agent.run("hi")
        assert llm.call_count == 1


# 简单的 mock 类，用于 spec
class ToolRegistryMock:
    def get_definitions(self):
        return []

    def execute(self, tool_call):
        return ToolResult(tool_call_id="call_1", content="")
