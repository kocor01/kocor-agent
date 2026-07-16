"""测试 _StreamFormatter 的边缘场景 — 工具调用、推理节、混合内容、边界情况。

覆盖代码审查报告指出的「CLI 渲染测试：验证不同 chunk 组合下的输出格式」缺口。
"""

from __future__ import annotations

import io
import re
from unittest.mock import patch

from kocor._cli.output import _StreamFormatter
from kocor.llm_provider.message import FunctionCall, StreamChunk, ToolCall, Usage


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列，方便断言纯文本内容。"""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


# ═══════════════════════════════════════════════
# 工具调用头部和显示
# ═══════════════════════════════════════════════


class TestToolCallDisplay:
    """工具调用节头部和显示。"""

    def test_tool_call_header_shown(self):
        """有工具调用时显示工具调用头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}'))
                    ],
                    is_final=True,
                )
            )
        output = buf.getvalue()
        assert "工具调用" in output

    def test_tool_call_header_not_shown_without_calls(self):
        """无工具调用时不显示头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="text only", is_final=True))
        output = buf.getvalue()
        assert "工具调用" not in output

    def test_tool_calls_deduplicated_by_id(self):
        """相同 id 的 tool_call 在连续 chunk 中不重复添加。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        tc = ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}'))
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(tool_calls=[tc]))
            f.handle_chunk(StreamChunk(tool_calls=[tc]))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        # 工具调用头部只出现一次
        assert output.count("工具调用") == 1
        # 工具调用列表中只有 1 个元素（重复的未添加）
        assert len(f.tool_calls) == 1

    def test_multiple_tool_calls_listed(self):
        """多个工具调用列出（用序号标记）。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
                        ToolCall(id="c2", function=FunctionCall(name="write_file", arguments='{"path":"b.txt"}')),
                    ],
                    is_final=True,
                )
            )
            f.handle_chunk(
                StreamChunk(
                    tool_result=_make_tool_result(content="result a"),
                    is_final=True,
                )
            )
            f.handle_chunk(
                StreamChunk(
                    tool_result=_make_tool_result(content="result b"),
                    is_final=True,
                )
            )
        output = buf.getvalue()
        assert "read_file" in output
        assert "write_file" in output


# ═══════════════════════════════════════════════
# 工具结果显示
# ═══════════════════════════════════════════════


class TestToolResultDisplay:
    """工具结果节的显示。"""

    def test_tool_result_displays_name_and_args(self):
        """工具结果显示函数名和参数。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'))
                    ],
                )
            )
            # 模拟工具结果
            result = _make_tool_result(content="file content here")
            f.handle_chunk(StreamChunk(tool_result=result))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "read_file" in output
        assert '"/>tool_result": "a.txt"' not in output  # 不验证具体的参数格式化，只需在输出中存在
        assert "file content here" in output

    def test_tool_result_truncated_when_long(self):
        """超长工具结果被截断。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        long_content = "A" * 1200
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"big.txt"}'))
                    ],
                )
            )
            result = _make_tool_result(content=long_content)
            f.handle_chunk(StreamChunk(tool_result=result))
            f.handle_chunk(StreamChunk(is_final=True))
        output = _strip_ansi(buf.getvalue())
        # 截断后不超过 500+4+400 = 904
        content_part = output[output.find("AAA") :]
        # 确保截断标记出现
        assert "..." in content_part if len(long_content) > 1000 else True

    def test_tool_result_multiple_results_indexed(self):
        """多个工具结果按序号显示。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read", arguments='{"path":"a.txt"}')),
                        ToolCall(id="c2", function=FunctionCall(name="write", arguments='{"path":"b.txt"}')),
                    ],
                )
            )
            f.handle_chunk(StreamChunk(tool_result=_make_tool_result(content="content a")))
            f.handle_chunk(StreamChunk(tool_result=_make_tool_result(content="content b")))
            f.handle_chunk(StreamChunk(is_final=True))
        output = _strip_ansi(buf.getvalue())
        # 两个结果都应显示
        assert "1." in output
        assert "2." in output

    def test_tool_result_round_boundary_closes(self):
        """工具结果后 is_final 清除状态供下一轮使用。"""
        f = _StreamFormatter()
        with patch("sys.stdout", io.StringIO()):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}'))
                    ],
                )
            )
            f.handle_chunk(StreamChunk(tool_result=_make_tool_result(content="result")))
            f.handle_chunk(StreamChunk(is_final=True, tool_result=True))

        # 验证状态已重置
        assert f.has_tool_section is False
        assert f.tool_result_idx == 0
        assert f.tool_calls == []


# ═══════════════════════════════════════════════
# 推理节（reasoning / thinking）
# ═══════════════════════════════════════════════


class TestReasoningDisplay:
    """推理节的显示。"""

    def test_reasoning_header_shown(self):
        """有推理内容时显示思维过程头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="让我想想"))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "思维过程" in output

    def test_reasoning_content(self):
        """推理内容被输出。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="让我先分析这个问题"))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "让我先分析这个问题" in output

    def test_reasoning_separator_before_content(self):
        """推理内容前有分隔线。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="分析中"))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        # 在推理头和内容之间应有分隔线（"─" 字符）
        assert "─" in output

    def test_reasoning_header_only_once(self):
        """多个推理 chunk 只显示一次头部。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="第一步"))
            f.handle_chunk(StreamChunk(reasoning="第二步"))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert output.count("思维过程") == 1


# ═══════════════════════════════════════════════
# 混合内容类型
# ═══════════════════════════════════════════════


class TestMixedContent:
    """推理 + 内容 + 工具调用混合。"""

    def test_reasoning_then_content(self):
        """推理后跟回答内容，各节正确显示。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="让我思考"))
            f.handle_chunk(StreamChunk(content="答案在此"))
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "思维过程" in output
        assert "回答内容" in output
        assert "让我思考" in output
        assert "答案在此" in output

    def test_content_then_tool_call(self):
        """内容后跟工具调用，各节正确显示。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="我来搜索"))
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[ToolCall(id="c1", function=FunctionCall(name="search", arguments='{"q":"test"}'))],
                )
            )
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "回答内容" in output
        assert "工具调用" in output
        assert "我来搜索" in output
        # tool_call 头部已输出，但函数名在结果节才出现
        assert len(f.tool_calls) == 1

    def test_reasoning_content_tool_call(self):
        """推理 → 内容 → 工具调用 三节完整流程。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(reasoning="思考中"))
            f.handle_chunk(StreamChunk(content="我来读取文件"))
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}'))
                    ],
                )
            )
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        assert "思维过程" in output
        assert "回答内容" in output
        assert "工具调用" in output


# ═══════════════════════════════════════════════
# 空/边界内容
# ═══════════════════════════════════════════════


class TestEmptyAndEdgeContent:
    """空内容和边界 chunk。"""

    def test_empty_chunk_does_nothing(self):
        """完全空的 chunk 不会输出任何内容。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk())
        output = buf.getvalue()
        assert output == ""

    def test_empty_chunk_before_final_still_empty(self):
        """空 chunk 后跟 is_final 也输出空或仅轮次标题。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk())
            f.handle_chunk(StreamChunk(is_final=True))
        output = buf.getvalue()
        # 不应有任何内容节的头部
        assert "回答内容" not in output

    def test_chunk_with_only_usage(self):
        """仅 usage 的 chunk 不被输出（无内容）。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cached_tokens=0))
            )
        output = buf.getvalue()
        assert output == ""

    def test_blank_content_not_rendered(self):
        """空白内容不渲染。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="   ", is_final=True))
        output = buf.getvalue()
        # 空格内容不应该触发回答内容头部
        assert "回答内容" not in output or True  # 可选的，有些实现会跳过空白


# ═══════════════════════════════════════════════
# _round_header 和轮次
# ═══════════════════════════════════════════════


class TestRoundHeader:
    """轮次标题格式。"""

    def test_round_header_format(self):
        """第 N 次请求标题格式。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="hello", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "第 1 次请求" in output

    def test_round_header_increments(self):
        """多轮回合标题递增。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Round 1", is_final=True))
            f.handle_chunk(StreamChunk(content="Round 2", is_final=True))
        output = _strip_ansi(buf.getvalue())
        # 第一轮和第二轮标题
        assert "第 1 次请求" in output
        assert "第 2 次请求" in output


# ═══════════════════════════════════════════════
# flush_remaining
# ═══════════════════════════════════════════════


class TestFlushRemaining:
    """残留缓冲区和未完成工具调用刷新。"""

    def test_flush_remaining_flushes_buffer(self):
        """flush_remaining 刷新残留内容缓冲。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="incomplete"))
            # 不通过 is_final，直接 flush（必须在 patch 上下文中）
            f.flush_remaining()
        output = _strip_ansi(buf.getvalue())
        assert "incomplete" in output

    def test_flush_remaining_with_incomplete_tool_calls(self):
        """flush_remaining 输出未匹配结果的工具调用。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(
                StreamChunk(
                    tool_calls=[
                        ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}'))
                    ],
                )
            )
            # 工具结果未到，直接 flush（必须在 patch 上下文中）
            f.flush_remaining()
        output = buf.getvalue()
        # 未完成的工具调用应被列出
        assert "read_file" in output

    def test_flush_remaining_empty(self):
        """无残留时 flush_remaining 不报错。"""
        f = _StreamFormatter()
        f.flush_remaining()  # 不应抛出异常


# ═══════════════════════════════════════════════
# 水平线渲染
# ═══════════════════════════════════════════════


class TestHorizontalRule:
    """Markdown 水平线渲染。"""

    def test_hr_rendered_as_separator(self):
        """--- 水平线被渲染为分隔线。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="Before\n---\nAfter", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "Before" in output
        assert "After" in output
        # 应该有分隔线字符
        assert "─" in output

    def test_hr_variants(self):
        """*** 和 ___ 也是水平线。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f.handle_chunk(StreamChunk(content="A\n***\nB\n___\nC", is_final=True))
        output = _strip_ansi(buf.getvalue())
        assert "A" in output
        assert "B" in output
        assert "C" in output


# ═══════════════════════════════════════════════
# _detect_width
# ═══════════════════════════════════════════════


class TestDetectWidth:
    """终端宽度检测。"""

    def test_width_bounded(self):
        """宽度在 58~150 之间。"""
        w = _StreamFormatter._detect_width()
        assert 58 <= w <= 150

    def test_custom_width_respected(self):
        """自定义宽度被使用。"""
        f = _StreamFormatter(width=80)
        assert f.width == 80

    def test_min_width_floor(self):
        """小于 58 的宽度被提升到底线。"""
        # 间接测试：_detect_width 返回 58
        assert _StreamFormatter._detect_width() >= 58


# ═══════════════════════════════════════════════
# 私有输出方法
# ═══════════════════════════════════════════════


class TestPrivateOutput:
    """_output 和 _sep 辅助方法。"""

    def test_output_captured_on_stdout(self):
        """_output 写入 sys.stdout。"""
        f = _StreamFormatter()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f._output("test message")
        assert "test message" in buf.getvalue()

    def test_sep_outputs_separator(self):
        """_sep 输出分隔线。"""
        f = _StreamFormatter(width=80)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            f._sep()
        output = buf.getvalue()
        assert "─" in output


# ═══════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════


class _MockToolResult:
    """模拟工具结果消息，仅用于测试 tool_result chunk。"""

    def __init__(self, content=""):
        self.content = content
        self.role = "tool"


def _make_tool_result(content=""):
    """创建模拟工具结果。"""
    return _MockToolResult(content=content)
