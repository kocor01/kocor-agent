"""轻量 Token 估算器。

使用启发式规则估算文本的 token 数，不引入 tiktoken 等外部依赖。
"""

from __future__ import annotations

import json

from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


def _is_cjk(char: str) -> bool:
    """判断字符是否为 CJK 字符（含全角标点、扩展区汉字等）。

    覆盖范围：
    - CJK Unified Ideographs (U+4E00-U+9FFF)
    - CJK Extension A (U+3400-U+4DBF)
    - CJK Extension B (U+20000-U+2A6DF)
    - CJK Extension C (U+2A700-U+2B73F)
    - CJK Symbols and Punctuation (U+3000-U+303F)
    - Fullwidth Forms (U+FF00-U+FFEF)
    """
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)   # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0x2A700 <= cp <= 0x2B73F)  # CJK Extension C
        or (0x3000 <= cp <= 0x303F)  # CJK Symbols and Punctuation
        or (0xFF00 <= cp <= 0xFFEF)  # Fullwidth Forms
    )


class TokenCounter:
    """Token 估算器。

    使用启发式规则估算：
    - 英文: ~4 chars / token
    - 中文: ~1.5 chars / token
    - 混合: 分别估算后相加

    设计决策：不引入 tiktoken 以保持轻量。
    启发式估算的误差约 ±20%，对预算管理来说足够。
    """

    ENGLISH_RATE = 4.0     # chars per token
    CHINESE_RATE = 1.5     # chars per token

    def count(self, text: str) -> int:
        """估算文本的 token 数。

        Args:
            text: 要估算的文本

        Returns:
            估算的 token 数，最小为 0
        """
        if not text:
            return 0

        chinese_chars = sum(1 for c in text if _is_cjk(c))
        ascii_chars = len(text) - chinese_chars

        token_estimate = (ascii_chars / self.ENGLISH_RATE) + (chinese_chars / self.CHINESE_RATE)
        return max(1, int(token_estimate))

    def count_message(self, message: Message) -> int:
        """估算单条消息的 token 数。

        除了 content 外，还计入 tool_calls、reasoning 和角色格式开销。

        Args:
            message: 要估算的消息

        Returns:
            估算的 token 数
        """
        total = self.count(message.content)

        if message.tool_calls:
            for tc in message.tool_calls:
                total += self.count(tc.function.name)
                total += self.count(tc.function.arguments)

        if message.reasoning:
            total += self.count(message.reasoning)

        # role 标记和格式开销
        total += 4

        return total

    def count_messages(self, messages: list[Message]) -> int:
        """估算消息列表的总 token 数。

        Args:
            messages: 消息列表

        Returns:
            估算的总 token 数
        """
        return sum(self.count_message(m) for m in messages)

    def count_tools(self, tools: list[ToolDefinition]) -> int:
        """估算工具定义列表的总 token 数。

        使用与 LLM API 请求相同的序列化格式估算。

        Args:
            tools: 工具定义列表

        Returns:
            估算的总 token 数
        """
        total = 0
        for tool in tools:
            serialized = json.dumps({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }, ensure_ascii=False, indent=2)
            total += self.count(serialized)
        return total
