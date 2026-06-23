"""测试 ToolOutputTruncator。"""

from __future__ import annotations

from kocor.context.truncator import ToolOutputTruncator
from kocor.llm_provider.message import Message


class TestToolOutputTruncator:
    """测试通用工具输出截断。"""

    def setup_method(self):
        self.truncator = ToolOutputTruncator()

    def test_short_text_not_truncated(self):
        result = self.truncator.truncate("hello world")
        assert result == "hello world"

    def test_empty_text_not_truncated(self):
        assert self.truncator.truncate("") == ""

    def test_line_too_long_truncated(self):
        """单行超长截断。"""
        long_line = "a" * 5000
        result = self.truncator.truncate(long_line)
        assert len(result) < 3000  # 被截断了
        assert "... [truncated]" in result

    def test_too_many_lines_truncated(self):
        """多行超行数截断。"""
        many_lines = "\n".join(f"line {i}" for i in range(5000))
        result = self.truncator.truncate(many_lines)
        lines = result.split("\n")
        # 应该被截断到 ~2000 行
        assert len(lines) < 3000

    def test_truncate_messages_empty(self):
        assert self.truncator.truncate_messages([]) == []

    def test_truncate_messages_short(self):
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]
        result = self.truncator.truncate_messages(msgs)
        assert len(result) == 2
        assert result[0].content == "你好"
        assert result[1].content == "你好！"

    def test_truncate_messages_long_tool_result(self):
        """超长 tool 消息应该被截断。"""
        long_content = "x" * 100000
        msgs = [
            Message(role="user", content="读文件"),
            Message(role="assistant", content="好的", tool_calls=[]),
            Message(role="tool", content=long_content, tool_call_id="call_1"),
        ]
        result = self.truncator.truncate_messages(msgs)
        # tool content 应该被截断
        assert len(result[2].content) < len(long_content)
        # user 和 assistant 消息不受影响
        assert result[0].content == "读文件"
        assert result[1].content == "好的"

    def test_truncate_messages_mixed_length(self):
        """混合长度消息，只截断过长的。"""
        normal = Message(role="user", content="正常文本")
        long_tool = Message(role="tool", content="a" * 60000, tool_call_id="call_1")
        medium = Message(role="assistant", content="中等长度文本" * 50)

        result = self.truncator.truncate_messages([normal, long_tool, medium])
        assert len(result) == 3
        assert result[0].content == "正常文本"
        assert len(result[1].content) < len(long_tool.content)
        assert result[2].content == medium.content

    def test_custom_config(self):
        """自定义截断配置。"""
        from kocor.mcp.truncate import TruncateConfig

        strict = TruncateConfig(max_bytes=100, max_lines=10, max_line_length=20)
        truncator = ToolOutputTruncator(config=strict)

        long_text = "hello world this is a long line\n" * 20
        result = truncator.truncate(long_text)
        # 应该被截断到 ~100 bytes
        assert len(result.encode("utf-8")) <= 200  # 余量，因为有截断标记
