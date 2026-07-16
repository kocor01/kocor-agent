"""测试 LLM Provider 的 _normalize_in/out 边界情况。

覆盖代码审查报告指出的「Provider 格式转换测试：_normalize_in/out 的边界情况
（空消息、多个 tool_calls、thinking 内容）」缺口。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kocor.config import Config
from kocor.llm_provider.message import FunctionCall, Message, ToolCall, Usage
from kocor.llm_provider.providers import AnthropicClient, OpenAIClient
from kocor.tools.definitions import ToolDefinition

# ═══════════════════════════════════════════════
# Anthropic _normalize_in 边界
# ═══════════════════════════════════════════════


class TestAnthropicNormalizeInEdgeCases:
    """Anthropic _normalize_in 边界。"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="anthropic")

    def test_empty_message_list(self):
        """空消息列表返回空。"""
        client = AnthropicClient()
        result = client._normalize_in([])
        assert result == []

    def test_tool_result_at_end_flushed(self):
        """末尾的 tool_result 被 flush。"""
        client = AnthropicClient()
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="tu_1", function=FunctionCall(name="read_file", arguments='{}')),
            ]),
            Message(role="tool", content="result", tool_call_id="tu_1"),
        ]
        result = client._normalize_in(messages)
        # 最后一条应是 user 消息（包含 tool_result）
        assert result[-1]["role"] == "user"
        assert result[-1]["content"][0]["type"] == "tool_result"

    def test_multiple_tool_results_merged(self):
        """多个连续 tool_result 合并到同一条 user 消息。"""
        client = AnthropicClient()
        messages = [
            Message(role="user", content="do things"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="tu_1", function=FunctionCall(name="read", arguments='{"p":"a"}')),
                ToolCall(id="tu_2", function=FunctionCall(name="write", arguments='{"p":"b"}')),
            ]),
            Message(role="tool", content="result a", tool_call_id="tu_1"),
            Message(role="tool", content="result b", tool_call_id="tu_2"),
        ]
        result = client._normalize_in(messages)
        # 验证两个 tool_result 在同一条消息中
        user_msgs = [m for m in result if m["role"] == "user"]
        # 最后一条 user 消息包含两个 tool_result
        last_user = user_msgs[-1]
        if isinstance(last_user["content"], list):
            assert len(last_user["content"]) == 2
            assert last_user["content"][0]["type"] == "tool_result"
            assert last_user["content"][1]["type"] == "tool_result"

    def test_tool_call_with_invalid_json_arguments(self):
        """工具调用的 arguments 为非法 JSON 时转换为空字典。"""
        client = AnthropicClient()
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="tu_1", function=FunctionCall(name="test", arguments="not valid json {")),
            ]),
        ]
        result = client._normalize_in(messages)
        assistant_msg = [m for m in result if m["role"] == "assistant"][-1]
        tool_use = assistant_msg["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["input"] == {}  # 非法 JSON 转为空字典

    def test_assistant_with_content_and_tool_calls(self):
        """assistant 消息同时有 content 和 tool_calls。"""
        client = AnthropicClient()
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="我来处理", tool_calls=[
                ToolCall(id="tu_1", function=FunctionCall(name="read", arguments='{"p":"a"}')),
            ]),
        ]
        result = client._normalize_in(messages)
        assistant_msg = [m for m in result if m["role"] == "assistant"][-1]
        content = assistant_msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "我来处理"
        assert content[1]["type"] == "tool_use"

    def test_user_with_empty_content(self):
        """空内容的 user 消息。"""
        client = AnthropicClient()
        messages = [Message(role="user", content="")]
        result = client._normalize_in(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == ""


# ═══════════════════════════════════════════════
# Anthropic _normalize_out 边界
# ═══════════════════════════════════════════════


class TestAnthropicNormalizeOutEdgeCases:
    """Anthropic _normalize_out 边界。"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="anthropic")

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_empty_content_blocks(self, mock_anthropic_cls):
        """空 content_blocks 的响应返回空字符串。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage.input_tokens = 0
        mock_response.usage.output_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="hi")])

        assert result.content == ""
        assert result.tool_calls == []

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_multiple_tool_calls_in_one_response(self, mock_anthropic_cls):
        """单个响应中多个工具调用。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_tool_block_1 = MagicMock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.id = "tu_1"
        mock_tool_block_1.name = "read_file"
        mock_tool_block_1.input = {"path": "a.txt"}

        mock_tool_block_2 = MagicMock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.id = "tu_2"
        mock_tool_block_2.name = "write_file"
        mock_tool_block_2.input = {"path": "b.txt", "content": "data"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block_1, mock_tool_block_2]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="do both")])

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "read_file"
        assert result.tool_calls[1].function.name == "write_file"

    @patch("kocor.llm_provider.providers.anthropic_client.Anthropic")
    def test_thinking_only_no_text(self, mock_anthropic_cls):
        """仅 thinking 无 text 块的响应。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "深入分析中..."

        mock_response = MagicMock()
        mock_response.content = [mock_thinking_block]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient()
        result = client.generate([Message(role="user", content="分析")])

        assert result.reasoning == "深入分析中..."
        assert result.content == ""
        assert result.tool_calls == []

    def test_normalize_tool(self):
        """_normalize_tool 格式转换。"""
        client = AnthropicClient()
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        result = client._normalize_tool(tool)
        assert result["name"] == "read_file"
        assert result["description"] == "Read a file"
        assert result["input_schema"] == tool.parameters


# ═══════════════════════════════════════════════
# OpenAI _normalize_in 边界
# ═══════════════════════════════════════════════


class TestOpenAINormalizeInEdgeCases:
    """OpenAI _normalize_in 边界。"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="openai")

    def test_empty_message_list(self):
        """空消息列表返回空。"""
        client = OpenAIClient()
        result = client._normalize_in([])
        assert result == []

    def test_multiple_roles(self):
        """多种 role 混合。"""
        client = OpenAIClient()
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="again"),
        ]
        result = client._normalize_in(messages)
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        """assistant 消息包含 tool_calls。"""
        client = OpenAIClient()
        messages = [
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
            ]),
        ]
        result = client._normalize_in(messages)
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_assistant_with_reasoning(self):
        """assistant 消息包含 reasoning。"""
        client = OpenAIClient()
        messages = [
            Message(role="assistant", content="结果", reasoning="让我想想"),
        ]
        result = client._normalize_in(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["reasoning"] == "让我想想"

    def test_tool_message(self):
        """tool 角色消息。"""
        client = OpenAIClient()
        messages = [
            Message(role="tool", content="file content", tool_call_id="call_1"),
        ]
        result = client._normalize_in(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "file content"

    def test_assistant_no_content_no_tool_calls(self):
        """assistant 消息无 content 且无 tool_calls。"""
        client = OpenAIClient()
        messages = [Message(role="assistant", content="")]
        result = client._normalize_in(messages)
        assert result[0]["role"] == "assistant"
        # content 不应出现在 body 中，因为 content 为空
        assert "content" not in result[0]

    def test_multiple_tool_calls_in_assistant(self):
        """单个 assistant 消息中包含多个 tool_calls。"""
        client = OpenAIClient()
        messages = [
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="c1", function=FunctionCall(name="read", arguments='{"p":"a"}')),
                ToolCall(id="c2", function=FunctionCall(name="write", arguments='{"p":"b"}')),
            ]),
        ]
        result = client._normalize_in(messages)
        assert len(result[0]["tool_calls"]) == 2


# ═══════════════════════════════════════════════
# OpenAI _normalize_out 边界
# ═══════════════════════════════════════════════


class TestOpenAINormalizeOutEdgeCases:

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="openai")

    def _make_response(self, mock_choice):
        """将 mock choice 包装为完整 response 对象。"""
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=None,
        )
        return mock_response

    def test_choice_without_tool_calls(self):
        """无 tool_calls 的 choice。"""
        client = OpenAIClient()

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning = None

        mock_response = self._make_response(mock_choice)
        result = client._normalize_out(mock_response)

        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.reasoning == ""

    def test_choice_with_reasoning(self):
        """带 reasoning 的 choice。"""
        client = OpenAIClient()

        mock_choice = MagicMock()
        mock_choice.message.content = "解析结果"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning = "这是推理过程"

        mock_response = self._make_response(mock_choice)
        result = client._normalize_out(mock_response)

        assert result.content == "解析结果"
        assert result.reasoning == "这是推理过程"

    def test_choice_with_tool_calls_and_reasoning(self):
        """同时有 tool_calls 和 reasoning 的 choice。"""
        client = OpenAIClient()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.type = "function"
        mock_tc.function.name = "read_file"
        mock_tc.function.arguments = '{"path": "a.txt"}'

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.message.reasoning = "读取文件..."

        mock_response = self._make_response(mock_choice)
        result = client._normalize_out(mock_response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "read_file"
        assert result.reasoning == "读取文件..."

    def test_multiple_tool_calls_in_choice(self):
        """单个 choice 中多个 tool_calls。"""
        client = OpenAIClient()

        mock_tc1 = MagicMock()
        mock_tc1.id = "c1"
        mock_tc1.type = "function"
        mock_tc1.function.name = "read"
        mock_tc1.function.arguments = '{"p":"a"}'

        mock_tc2 = MagicMock()
        mock_tc2.id = "c2"
        mock_tc2.type = "function"
        mock_tc2.function.name = "write"
        mock_tc2.function.arguments = '{"p":"b"}'

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tc1, mock_tc2]
        mock_choice.message.reasoning = None

        mock_response = self._make_response(mock_choice)
        result = client._normalize_out(mock_response)

        assert len(result.tool_calls) == 2

    def test_choice_with_usage(self):
        """传递 usage 信息。"""
        client = OpenAIClient()

        mock_choice = MagicMock()
        mock_choice.message.content = "done"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning = None

        mock_response = self._make_response(mock_choice)
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cached_tokens=5)
        result = client._normalize_out(mock_response, usage=usage)

        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.cached_tokens == 5

    def test_empty_content(self):
        """空内容的 choice。"""
        client = OpenAIClient()

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning = None

        mock_response = self._make_response(mock_choice)
        result = client._normalize_out(mock_response)

        assert result.content == ""


# ═══════════════════════════════════════════════
# 测试 _to_openai_tool
# ═══════════════════════════════════════════════


class TestOpenAIToolConversion:
    """_normalize_tool 工具格式转换。"""

    def test_normalize_tool_format(self):
        client = OpenAIClient()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        result = client._normalize_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"
        assert result["function"]["parameters"]["properties"]["x"]["type"] == "string"


# ═══════════════════════════════════════════════
# 测试 Anthropic _extract_system
# ═══════════════════════════════════════════════


class TestAnthropicPrepareMessages:
    """_prepare_messages 边界。"""

    def setup_method(self):
        Config.reset()
        Config._instance = Config(provider="anthropic")

    def test_no_system_messages(self):
        """无 system 消息时返回 None 和原列表。"""
        client = AnthropicClient()
        messages = [Message(role="user", content="hi")]
        system, filtered = client._prepare_messages(messages)
        assert system is None
        assert len(filtered) == 1

    def test_multiple_system_messages_joined(self):
        """多个 system 消息用分隔符拼接。"""
        client = AnthropicClient()
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="hi"),
            Message(role="system", content="[摘要] 历史摘要"),
        ]
        system, filtered = client._prepare_messages(messages)
        assert "你是助手" in system
        assert "历史摘要" in system
        assert "---" in system
        assert all(m.role != "system" for m in filtered)

    def test_system_content_is_empty(self):
        """空内容的 system 消息被忽略。"""
        client = AnthropicClient()
        messages = [
            Message(role="system", content=""),
            Message(role="user", content="hi"),
        ]
        system, filtered = client._prepare_messages(messages)
        assert system is None
        assert len(filtered) == 1