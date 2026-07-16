"""StreamSession 单元测试。

验证流式 LLM 响应的消费和轮次边界管理。
"""

from __future__ import annotations

from typing import Iterator

from kocor._stream_session import StreamSession
from kocor.llm_provider.message import StreamChunk as Chunk


def _stream(*chunks: Chunk) -> Iterator[Chunk]:
    yield from chunks


def _tc(id_: str, name: str, args: dict | None = None) -> object:
    """辅助创建 ToolCall-like 对象。"""
    from unittest.mock import MagicMock
    tc = MagicMock()
    tc.id = id_
    tc.function.name = name
    tc.function.arguments = args or {}
    return tc


class TestStreamSession:
    """StreamSession 基础功能测试。"""

    def test_accumulates_text_content(self):
        """流式文本应被正确累积。"""
        sess = StreamSession(_stream(
            Chunk(content="Hello "),
            Chunk(content="World"),
            Chunk(is_final=True),
        ))
        list(sess.iter_chunks())  # 消费所有块
        assert sess.message().content == "Hello World"

    def test_accumulates_reasoning(self):
        """reasoning 内容应被正确累积。"""
        sess = StreamSession(_stream(
            Chunk(reasoning="思考"),
            Chunk(reasoning="过程"),
            Chunk(is_final=True),
        ))
        list(sess.iter_chunks())
        assert sess.message().reasoning == "思考过程"

    def test_no_tool_calls(self):
        """纯文本回复时 has_tool_calls 为 False。"""
        sess = StreamSession(_stream(
            Chunk(content="Hello"),
            Chunk(is_final=True),
        ))
        list(sess.iter_chunks())
        assert not sess.has_tool_calls

    def test_accumulates_tool_calls(self):
        """多个工具调用应被正确累积。"""
        sess = StreamSession(_stream(
            Chunk(tool_calls=[_tc("1", "read", {"path": "/a"})]),
            Chunk(tool_calls=[_tc("2", "write", {"path": "/b"})]),
            Chunk(is_final=True),
        ))
        list(sess.iter_chunks())
        assert sess.has_tool_calls
        assert len(sess.message().tool_calls) == 2

    def test_deduplicates_tool_calls(self):
        """同 id 的 tool_call 只保留一个。"""
        sess = StreamSession(_stream(
            Chunk(tool_calls=[_tc("1", "read", {"path": "/a"})]),
            Chunk(tool_calls=[_tc("1", "read", {"path": "/a"})]),  # 重复
            Chunk(is_final=True),
        ))
        list(sess.iter_chunks())
        assert len(sess.message().tool_calls) == 1

    def test_absorbs_final_marker(self):
        """纯结束标记被吸收，不透传给迭代器。"""
        sess = StreamSession(_stream(
            Chunk(content="hello"),
            Chunk(is_final=True),  # 无内容无语义
        ))
        visible = list(sess.iter_chunks())
        # 只有 "hello" 块是可见的
        assert len(visible) == 1
        assert visible[0].content == "hello"

    def test_passes_content_with_final(self):
        """含内容的 is_final 应透传。"""
        sess = StreamSession(_stream(
            Chunk(content="done", is_final=True),
        ))
        visible = list(sess.iter_chunks())
        assert len(visible) == 1
        assert visible[0].content == "done"
        assert visible[0].is_final

    def test_request_stop_halts_consumption(self):
        """request_stop 后流被截断。"""
        sess = StreamSession(_stream(
            Chunk(content="part1"),
            Chunk(content="part2"),
            Chunk(content="part3"),
        ))
        it = iter(sess.iter_chunks())
        assert next(it).content == "part1"
        sess.request_stop()
        remaining = list(it)
        # 停止后可能还有 0 或 1 个块，但不能更多
        assert len(remaining) <= 1

    def test_message_after_empty_stream(self):
        """空流返回空 Message。"""
        sess = StreamSession(_stream())
        list(sess.iter_chunks())
        msg = sess.message()
        assert msg.content == ""
        assert not msg.tool_calls
        assert msg.usage is None

    def test_iter_chunks_yields_visible_chunks(self):
        """iter_chunks 应产出可见块并按顺序排列。"""
        sess = StreamSession(_stream(
            Chunk(content="a"),
            Chunk(content="b", tool_calls=[_tc("1", "read")]),
            Chunk(content="c"),
            Chunk(is_final=True),
        ))
        visible = list(sess.iter_chunks())
        assert len(visible) == 3
        assert visible[0].content == "a"
        assert visible[1].content == "b"
        assert visible[2].content == "c"

    def test_usage_accumulates(self):
        """usage 应被正确累积。"""
        from kocor.llm_provider.message import Usage
        sess = StreamSession(_stream(
            Chunk(content="text", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cached_tokens=0)),
        ))
        list(sess.iter_chunks())
        assert sess.message().usage.prompt_tokens == 10
        assert sess.message().usage.completion_tokens == 5