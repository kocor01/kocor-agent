"""上下文管理数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kocor.context.budget import TokenBudget
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class ContextStrategy(Enum):
    """上下文管理策略。

    DEFAULT: 全量消息，无截断（适合短会话）
    SLIDING_WINDOW: 保留最开始 N 轮 + 摘要中间轮次 + 保留最近 N 轮完整消息
    AGGRESSIVE: 仅保留最后一轮完整对话 + 其余历史摘要
    """

    DEFAULT = "default"
    SLIDING_WINDOW = "sliding"
    AGGRESSIVE = "aggressive"


@dataclass
class MemoryItem:
    """单条持久记忆。

    Attributes:
        name: 唯一标识名（用作文件名 slug）
        description: 一行摘要，用于检索时判断相关性
        content: 记忆内容（Markdown 文本）
        memory_type: 类型（user / feedback / project / reference）
        created_at: 创建时间 ISO 格式
        updated_at: 最后更新时间 ISO 格式
    """

    name: str
    description: str
    content: str
    memory_type: str = "reference"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SummaryNode:
    """摘要节点，代表一段被压缩的历史。

    Attributes:
        summary: 摘要文本
        message_count: 原始消息数
        token_count: 摘要后 token 数
        original_start: 原始消息起始索引
        original_end: 原始消息结束索引
    """

    summary: str
    message_count: int
    token_count: int
    original_start: int
    original_end: int


@dataclass
class AgentContext:
    """Agent 运行时上下文，包含构建最终 prompt 的所有信息。

    Attributes:
        system_content: L1-L4 合并后的系统提示文本
        tool_definitions: 可用工具定义（L6）
        session_messages: 当前会话消息列表（最终发送给 LLM 的 messages）
        session_memory: 会话级 KV 记忆（仅当前 session 有效）
        token_budget: Token 预算与使用统计
    """

    system_content: str
    tool_definitions: list[ToolDefinition]
    session_messages: list[Message]
    session_memory: dict[str, str] = field(default_factory=dict)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
