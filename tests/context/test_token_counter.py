"""测试 TokenCounter。"""

from __future__ import annotations

from kocor.context.token_counter import TokenCounter
from kocor.llm_provider.message import FunctionCall, Message, ToolCall


class TestTokenCounter:
    """测试 TokenCounter 启发式 token 估算。"""

    def setup_method(self):
        self.counter = TokenCounter()

    def test_empty_string(self):
        assert self.counter.count("") == 0

    def test_null_string(self):
        assert self.counter.count("") == 0

    def test_english_word(self):
        """英文 4 字符约等于 1 token。"""
        n = self.counter.count("hello")
        assert n >= 1
        assert n <= 2

    def test_english_sentence(self):
        text = "hello world this is a test"
        n = self.counter.count(text)
        # 25 chars / 4 = 6.25
        assert 5 <= n <= 10

    def test_chinese_chars(self):
        """中文 1.5 字符约等于 1 token。"""
        text = "你好世界"
        n = self.counter.count(text)
        # 4 个中文字符 / 1.5 = 2.67
        assert 2 <= n <= 4

    def test_chinese_long_text(self):
        text = "你好，这是一个测试文本，用于验证中文 Token 估算的准确性"
        n = self.counter.count(text)
        assert n > 5

    def test_mixed_chinese_and_english(self):
        text = "你好 world 测试 hello"
        n = self.counter.count(text)
        # ascii: 11 / 4 = 2.75, chinese: 6 / 1.5 = 4
        # total ~= 6.75
        assert 4 <= n <= 10

    def test_count_message_plain_text(self):
        msg = Message(role="assistant", content="你好")
        n = self.counter.count_message(msg)
        assert n >= 2  # content + role overhead

    def test_count_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                ),
            ],
        )
        n = self.counter.count_message(msg)
        assert n > 0

    def test_count_message_with_reasoning(self):
        msg = Message(
            role="assistant",
            content="最终答案",
            reasoning="思考过程...",
        )
        n = self.counter.count_message(msg)
        assert n > 0

    def test_count_messages_empty(self):
        assert self.counter.count_messages([]) == 0

    def test_count_messages_multiple(self):
        msgs = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！有什么可以帮助你的？"),
        ]
        n = self.counter.count_messages(msgs)
        assert n > 0
        assert n > self.counter.count_message(msgs[0])