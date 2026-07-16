"""流式 LLM 响应的消费和轮次边界管理。

StreamSession 封装单次 LLM 流式调用的消费过程：
- 累积 content、reasoning、tool_calls、usage
- 吸收 LLM 流自带的 is_final 标记（不透传给渲染层）
- 支持外部中断（request_stop）
- 产出合并后的 Message
"""

from __future__ import annotations

from typing import Iterator

from kocor.llm_provider.message import Message, StreamChunk, ToolCall, Usage


class StreamSession:
    """管理单次 LLM 流式响应的消费。

    用法:
        sess = StreamSession(llm.stream(...))
        for visible_chunk in sess.iter_chunks():
            yield visible_chunk
        response = sess.message()  # 合并后的完整 Message
    """

    def __init__(self, llm_stream: Iterator[StreamChunk]):
        self._stream = llm_stream

        # 累积状态
        self._accumulated_tool_calls: list[ToolCall] = []
        self._final_content = ""
        self._final_reasoning = ""
        self._usage: Usage | None = None

        # 中断标志
        self._is_stopped = False

    def request_stop(self) -> None:
        """请求中断流式消费。"""
        self._is_stopped = True

    @property
    def is_stopped(self) -> bool:
        return self._is_stopped

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._accumulated_tool_calls)

    def iter_chunks(self) -> Iterator[StreamChunk]:
        """消费 LLM 流，产出可见块。

        LLM 流自带的纯结束标记（is_final 且无实质内容）被吸收，
        由循环层统一管控轮次边界。
        """
        for chunk in self._stream:
            if self._is_stopped:
                break

            # 累积工具调用（按 id 去重）
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    if not any(t.id == tc.id for t in self._accumulated_tool_calls):
                        self._accumulated_tool_calls.append(tc)

            # 累积文本
            if chunk.content:
                self._final_content += chunk.content

            # 累积 reasoning
            if chunk.reasoning:
                self._final_reasoning += chunk.reasoning

            # 累积 usage
            if chunk.usage:
                self._usage = chunk.usage

            # 吸收纯结束标记（无内容、无工具调用的 is_final）
            if chunk.is_final and not chunk.content and not chunk.reasoning and not chunk.tool_calls:
                continue

            yield chunk

    def message(self) -> Message:
        """返回累积后的完整 Message。"""
        return Message(
            role="assistant",
            content=self._final_content,
            reasoning=self._final_reasoning,
            tool_calls=self._accumulated_tool_calls or None,
            usage=self._usage,
        )