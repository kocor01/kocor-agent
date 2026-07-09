"""测试 _StreamFormatter 的 Markdown 渲染功能。"""

from __future__ import annotations

import io
import re
from unittest.mock import patch

import pytest

from kocor.cli import _StreamFormatter
from kocor.llm_provider.message import StreamChunk


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列，方便断言纯文本内容。"""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


class TestMarkdownBasic:
    """基础 Markdown 渲染。"""

    def test_plain_text(self):
        """纯文本内容正常显示。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Hello World", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Hello World" in output

    def test_bold_text(self):
        """**bold** 标记被正确解析去除。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Hello **World**", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Hello" in output
        assert "World" in output
        # ** 语法标记应被 Rich 去除
        assert "**World**" not in output

    def test_inline_code(self):
        """`code` 标记被正确解析去除。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Use `print()` to output.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "print()" in output
        # `` 语法标记应被 Rich 去除
        assert "`print()`" not in output


class TestParagraphBoundary:
    """段落边界分块渲染。"""

    def test_two_paragraphs(self):
        """双段落正确输出。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="First para.\n\nSecond para.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "First para." in output
        assert "Second para." in output

    def test_content_split_across_chunks(self):
        """半段落在 chunk 间累积后正确渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="First paragraph.\n\nSecond "))
            f.handle_chunk(StreamChunk(content="paragraph.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "First paragraph." in output
        assert "Second paragraph." in output

    def test_single_newline_within_paragraph(self):
        """段落内的单个换行不触发分段。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Line 1\nLine 2\n\nNext para", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Next para" in output

    def test_multiple_chunks_no_paragraph_boundary(self):
        """多个 chunk 无段落边界时只最终渲染一次。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Hello "))
            f.handle_chunk(StreamChunk(content="World "))
            f.handle_chunk(StreamChunk(content="!!!", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Hello World !!!" in output

    def test_triple_paragraphs(self):
        """三个段落拆分到三个 chunk。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Para 1\n\n"))
            f.handle_chunk(StreamChunk(content="Para 2\n\n"))
            f.handle_chunk(StreamChunk(content="Para 3", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Para 1" in output
        assert "Para 2" in output
        assert "Para 3" in output

    def test_leading_newlines_are_stripped(self):
        """内容前缀空行被去除。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="\n\n\nHello", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Hello" in output

    def test_empty_content(self):
        """空内容不显示回答内容头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="", is_final=True))
        output = buf.getvalue()
        # 轮次标题可能仍会输出，但回答内容头部不应出现
        assert "回答内容" not in output


class TestLineLevel:
    """行级实时渲染。"""

    def test_lines_rendered_as_arrive(self):
        """每收到一个完整行立即渲染，而非等到段落结束。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Line 1\nLine 2\nLine 3", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output

    def test_incomplete_line_buffered(self):
        """不带换行的不完整行保持缓冲，直到下一个换行或结束时刷新。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Hello "))
            f.handle_chunk(StreamChunk(content="World\nNext line.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Hello World" in output
        assert "Next line." in output

    def test_consecutive_newlines_produce_blank_lines(self):
        """连续换行符之间的空行被跳过。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="A\n\n\nB", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "A" in output
        assert "B" in output

    def test_bold_on_same_line(self):
        """同一行内的粗体标记正确渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="This is **bold** text.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "This is bold text." in output
        assert "**bold**" not in output

    def test_code_block_line_by_line(self):
        """代码块中的行流式到达，但整块累积后在闭合时一次性渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="```python\nprint(1)\nprint(2)\n```\nDone.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "print(1)" in output
        assert "print(2)" in output
        assert "Done." in output
        # ``` 标记不应出现在输出中
        assert "```" not in _strip_ansi(buf.getvalue())


class TestCodeBlockSmart:
    """代码块智能批处理。"""

    def test_code_block_cross_chunks(self):
        """代码块跨多个 chunk 到达，闭合后整块渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="```python\n"))
            f.handle_chunk(StreamChunk(content="print(1)\n"))
            f.handle_chunk(StreamChunk(content="print(2)\n"))
            f.handle_chunk(StreamChunk(content="```\n"))
            f.handle_chunk(StreamChunk(content="Done.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "print(1)" in output
        assert "print(2)" in output
        assert "Done." in output

    def test_code_block_unclosed_flush_on_final(self):
        """未闭合的代码块在 is_final 时强制刷新。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="```python\nprint(1)\nprint(2)", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "print(1)" in output
        assert "print(2)" in output

    def test_mixed_inline_and_code(self):
        """普通文本行级渲染 + 代码块批处理混合工作。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Before.\n```\ncode\n```\nAfter.", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Before." in output
        assert "code" in output
        assert "After." in output


class TestCodeBlock:
    """代码块渲染。"""

    def test_code_block(self):
        """Markdown 代码块被渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(
                content="Example:\n\n```python\nprint('hello')\n```\n\nDone.",
                is_final=True,
            ))
        output = _strip_ansi(buf.getvalue())
        assert "Example:" in output
        assert "print('hello')" in output
        assert "Done." in output


class TestHeaderDisplay:
    """回答内容头部的显示逻辑。"""

    def test_header_shown_with_content(self):
        """有内容时显示回答内容头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Hello", is_final=True))
        output = buf.getvalue()
        assert "回答内容" in output

    def test_header_not_shown_without_content(self):
        """无内容时不显示头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "回答内容" not in output


class TestRoundReset:
    """多轮对话时状态重置。"""

    def test_two_rounds_content_rendered(self):
        """两轮内容分别渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            # 第一轮
            f.handle_chunk(StreamChunk(content="Round 1", is_final=True))
            # 第二轮（新的非最终 chunk 会触新一轮开始）
            f.handle_chunk(StreamChunk(content="Round 2", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Round 1" in output
        assert "Round 2" in output


class TestTableRendering:
    """Markdown 表格渲染。"""

    def test_table_renders_as_single_block(self):
        """表格各行列（| 开头）应整体渲染，不分行割裂。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="| 城市 | 天气 | 温度 |\n", is_final=False))
            f.handle_chunk(StreamChunk(content="|------|------|------|\n", is_final=False))
            f.handle_chunk(StreamChunk(content="| 北京 | ☀️ 晴 | 25°C |\n", is_final=False))
            f.handle_chunk(StreamChunk(content="| 上海 | 🌧️ 雨 | 22°C |\n", is_final=False))
            f.handle_chunk(StreamChunk(content="", is_final=True))
        output = _strip_ansi(buf.getvalue())
        # 验证所有表格内容都在输出中
        assert "北京" in output
        assert "上海" in output
        assert "25°C" in output
        assert "22°C" in output

    def test_mixed_paragraph_and_table(self):
        """段落文字应以流式即时渲染，表格行应累积渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="天气情况：\n\n", is_final=False))
            f.handle_chunk(StreamChunk(content="| 城市 | 温度 |\n", is_final=False))
            f.handle_chunk(StreamChunk(content="|------|------|\n", is_final=False))
            f.handle_chunk(StreamChunk(content="| 北京 | 25°C |\n", is_final=False))
            f.handle_chunk(StreamChunk(content="\n\n", is_final=False))
            f.handle_chunk(StreamChunk(content="以上就是天气数据。", is_final=True))
        output = _strip_ansi(buf.getvalue())
        # 段落内容和表格内容都应在输出中
        assert "天气情况" in output
        assert "北京" in output
        assert "25°C" in output
        assert "以上就是天气数据" in output


class TestConsoleCaching:
    """_console 不应在类级别缓存 Console 实例。"""

    def test_console_not_cached_at_class_level(self):
        """类级别缓存可能导致状态泄漏，每次调用应返回新实例。"""
        c1 = _StreamFormatter._console()
        c2 = _StreamFormatter._console()
        assert c1 is not c2, "_console() 不应返回同一个缓存的 Console 实例"

    def test_separator_uses_current_stdout(self):
        """分隔线应写入当前 sys.stdout。"""
        f = _StreamFormatter(width=80)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="testing reasoning"))
        output = buf.getvalue()
        assert "─" in output, "分隔线应被捕获在当前 patch 的 stdout 中"
        assert "testing reasoning" in output
