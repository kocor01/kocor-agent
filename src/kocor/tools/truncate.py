from __future__ import annotations

from kocor.llm_provider.message import Message


class ToolOutputTruncator:
    """工具输出截断器。

    对工具输出执行三级截断策略：
    1. 单行截断 — 超长行末尾截断
    2. 行数截断 — 超出保留行数的部分头尾各保留 50%
    3. 字节截断 — 超出最大字节数的部分头尾各保留 50%
    """

    def __init__(
        self,
        max_bytes: int = 50_000,
        max_lines: int = 2_000,
        max_line_length: int = 2_000,
    ):
        self.max_bytes = max_bytes
        self.max_lines = max_lines
        self.max_line_length = max_line_length

    def truncate(self, text: str, tool_name: str = "") -> str:
        """对工具输出执行三级截断。

        Args:
            text: 原始输出文本
            tool_name: 工具名称（当前仅用于日志，不影响截断逻辑）

        Returns:
            截断后的文本
        """
        if not text:
            return text

        # 阶段 1：单行截断
        lines = text.split("\n")
        truncated_lines = []
        for line in lines:
            if len(line) > self.max_line_length:
                line = line[:self.max_line_length] + "... [truncated]"
            truncated_lines.append(line)

        # 阶段 2：行数截断
        if len(truncated_lines) > self.max_lines:
            head_count = int(self.max_lines * 0.5)
            tail_count = self.max_lines - head_count
            head = truncated_lines[:head_count]
            tail = truncated_lines[-tail_count:]
            omitted = len(truncated_lines) - self.max_lines
            truncated_lines = head + [
                f"\n[... {omitted} lines truncated ...]\n"
            ] + tail

        result = "\n".join(truncated_lines)

        # 阶段 3：字节截断
        if len(result.encode("utf-8")) > self.max_bytes:
            half = self.max_bytes // 2
            head = result[:half]
            tail = result[-half:]
            result = f"{head}\n[... OUTPUT TRUNCATED ...]\n{tail}"

        return result

    def truncate_messages(self, messages: list[Message]) -> list[Message]:
        """对消息列表中的 tool 消息执行截断。

        非 tool 消息不做处理直接返回。

        Args:
            messages: 消息列表

        Returns:
            截断后的消息列表
        """
        result = []
        for msg in messages:
            if msg.role == "tool" and msg.content:
                truncated = self.truncate(msg.content)
                if truncated != msg.content:
                    msg = Message(
                        role=msg.role,
                        content=truncated,
                        tool_call_id=msg.tool_call_id,
                    )
            result.append(msg)
        return result