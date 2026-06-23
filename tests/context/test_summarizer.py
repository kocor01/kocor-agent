"""测试 HistorySummarizer。"""

from __future__ import annotations

from kocor.context.models import SummaryNode
from kocor.context.summarizer import HistorySummarizer
from kocor.llm_provider.message import FunctionCall, Message, ToolCall
from kocor.tools.definitions import ToolDefinition


class FakeLLMForSummary:
    """伪造的 LLM 客户端，始终返回预设的摘要文本。"""

    def __init__(self, summary_text: str = "这是对话摘要"):
        self.summary_text = summary_text
        self.call_count = 0
        self.last_messages = None

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        self.call_count += 1
        self.last_messages = messages
        return Message(role="assistant", content=self.summary_text)

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        raise NotImplementedError


class TestHistorySummarizer:
    """测试 HistorySummarizer。"""

    def test_summarize_returns_summary_node(self):
        """summarize() 应返回 SummaryNode。"""
        llm = FakeLLMForSummary()
        summarizer = HistorySummarizer(llm=llm)
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]
        node = summarizer.summarize(msgs)
        assert isinstance(node, SummaryNode)
        assert node.summary == "这是对话摘要"
        assert node.message_count == 2

    def test_summarize_calls_llm(self):
        """summarize() 应调用 LLM。"""
        llm = FakeLLMForSummary()
        summarizer = HistorySummarizer(llm=llm)
        summarizer.summarize([
            Message(role="user", content="问题1"),
            Message(role="assistant", content="回答1"),
        ])
        assert llm.call_count == 1

    def test_summarize_token_count_estimated(self):
        """摘要的 token_count 应被估算。"""
        llm = FakeLLMForSummary(summary_text="短摘要")
        summarizer = HistorySummarizer(llm=llm)
        node = summarizer.summarize([
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])
        assert node.token_count > 0

    def test_summarize_empty_list(self):
        """空消息列表应返回空摘要。"""
        llm = FakeLLMForSummary()
        summarizer = HistorySummarizer(llm=llm)
        node = summarizer.summarize([])
        assert node.summary == ""
        assert node.message_count == 0
        assert node.token_count == 0
        # LLM 不应被调用
        assert llm.call_count == 0

    def test_summarize_captures_indexes(self):
        """摘要应记录原始消息索引。"""
        llm = FakeLLMForSummary()
        summarizer = HistorySummarizer(llm=llm)
        msgs = [Message(role="user", content=str(i)) for i in range(5)]
        node = summarizer.summarize(msgs, start_index=2, end_index=7)
        assert node.original_start == 2
        assert node.original_end == 7

    def test_summarize_formats_messages_with_tool_calls(self):
        """消息包含工具调用时应被正确格式化。"""
        llm = FakeLLMForSummary()
        summarizer = HistorySummarizer(llm=llm)
        msgs = [
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
            ]),
            Message(role="tool", content="file content", tool_call_id="c1"),
        ]
        node = summarizer.summarize(msgs)
        assert node.message_count == 2
        assert node.summary != ""

    def test_custom_summarization_prompt(self):
        """自定义摘要 prompt。"""
        llm = FakeLLMForSummary()
        custom_prompt = "请用中文总结：{history_text}"
        summarizer = HistorySummarizer(llm=llm, summarization_prompt=custom_prompt)
        summarizer.summarize([
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])
        # 验证自定义 prompt 被使用
        assert llm.last_messages is not None
        last_msg = llm.last_messages[0]
        assert "请用中文总结" in last_msg.content