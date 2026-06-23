"""通用工具输出截断。

适用于所有工具输出的截断（不仅仅 MCP 工具），
包装 mcp/truncate.py 的现有三级截断策略。
"""

from __future__ import annotations

from kocor.llm_provider.message import Message
from kocor.mcp.truncate import TruncateConfig, truncate_output


class ToolOutputTruncator:
    """工具输出截断器。

    包装现有的三级截断策略，适用于所有工具输出：

    1. 单行截断 — 超长行末尾截断
    2. 行数截断 — 超出保留行数的部分头尾各保留 50%
    3. 字节截断 — 超出最大字节数的部分头尾各保留 50%

    Attributes:
        config: 截断配置
    """

    DEFAULT_CONFIG = TruncateConfig(
        max_bytes=50_000,
        max_lines=2_000,
        max_line_length=2_000,
    )

    def __init__(self, config: TruncateConfig | None = None):
        self.config = config or self.DEFAULT_CONFIG

    def truncate(self, text: str, tool_name: str = "") -> str:
        """对工具输出执行三级截断。

        Args:
            text: 原始输出文本
            tool_name: 工具名称（当前仅用于日志，不影响截断逻辑）

        Returns:
            截断后的文本
        """
        return truncate_output(text, self.config)

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
