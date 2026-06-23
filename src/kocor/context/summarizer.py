"""会话历史摘要器。

使用 LLM 将多轮对话压缩为一段摘要，保留关键信息。
"""

from __future__ import annotations

from kocor.context.models import SummaryNode
from kocor.context.token_counter import TokenCounter
from kocor.llm_provider.message import Message

DEFAULT_SUMMARIZATION_PROMPT = """\
请压缩以下对话为一段摘要，保留所有关键信息（包括用户需求、工具调用结果、重要上下文）。
摘要应该简洁但完整，以便后续理解对话背景。

对话内容：
{history_text}"""


class HistorySummarizer:
    """会话历史摘要器。

    使用 LLM 将一段消息列表压缩为文本摘要。

    Attributes:
        llm: LLM 客户端，用于生成摘要
        summarization_prompt: 摘要 prompt 模板，包含 {history_text} 占位符
        max_input_chars: 摘要输入的最大字符数（防止输入过长）
    """

    def __init__(
        self,
        llm,
        summarization_prompt: str | None = None,
        max_input_chars: int = 10_000,
    ):
        self.llm = llm
        self.summarization_prompt = summarization_prompt or DEFAULT_SUMMARIZATION_PROMPT
        self.max_input_chars = max_input_chars
        self._token_counter = TokenCounter()

    def summarize(
        self,
        messages: list[Message],
        start_index: int = 0,
        end_index: int = 0,
    ) -> SummaryNode:
        """将消息列表压缩为摘要。

        Args:
            messages: 要摘要的消息列表
            start_index: 原始消息起始索引（用于记录位置）
            end_index: 原始消息结束索引（用于记录位置）

        Returns:
            SummaryNode: 摘要节点
        """
        if not messages:
            return SummaryNode(
                summary="",
                message_count=0,
                token_count=0,
                original_start=start_index,
                original_end=end_index,
            )

        # 将消息格式化为文本
        history_text = self._messages_to_text(messages)

        # 截断过长的输入（防止 token 溢出）
        if len(history_text) > self.max_input_chars:
            history_text = history_text[:self.max_input_chars] + "\n[... 截断 ...]"

        # 调用 LLM 生成摘要
        prompt = self.summarization_prompt.format(history_text=history_text)
        msg = Message(role="user", content=prompt)
        result = self.llm.generate([msg])

        return SummaryNode(
            summary=result.content,
            message_count=len(messages),
            token_count=self._token_counter.count(result.content),
            original_start=start_index,
            original_end=end_index if end_index > start_index else len(messages),
        )

    def _messages_to_text(self, messages: list[Message]) -> str:
        """将消息列表格式化为纯文本。"""
        lines = []
        for msg in messages:
            role_label = {
                "user": "用户",
                "assistant": "助手",
                "tool": "工具结果",
                "system": "系统",
            }.get(msg.role, msg.role)

            lines.append(f"[{role_label}]")

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"  -> 调用工具: {tc.function.name}({tc.function.arguments})")

            if msg.content:
                # 截断超长内容，避免工具结果膨胀
                content = msg.content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                lines.append(f"  {content}")

        return "\n".join(lines)