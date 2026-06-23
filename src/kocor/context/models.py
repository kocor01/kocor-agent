"""上下文管理数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from kocor.llm_provider.message import Message
from kocor.llm_provider.tool_definition import ToolDefinition


@dataclass
class TokenBudget:
    """Token 预算与使用统计。

    Attributes:
        limit: 上下文窗口上限 token 数
        used_prompt: 当前 prompt 已用 token
        used_completion: 当前 completion 已用 token
        threshold_summary: 触发摘要的阈值比例（0.0 ~ 1.0）
        threshold_truncate: 触发截断的阈值比例（0.0 ~ 1.0）
    """

    limit: int = 200_000
    used_prompt: int = 0
    used_completion: int = 0
    threshold_summary: float = 0.70
    threshold_truncate: float = 0.90

    @property
    def remaining(self) -> int:
        return self.limit - self.used_prompt

    @property
    def usage_ratio(self) -> float:
        if self.limit <= 0:
            return 0.0
        return self.used_prompt / self.limit

    def should_summarize(self) -> bool:
        return self.usage_ratio >= self.threshold_summary

    def should_truncate(self) -> bool:
        return self.usage_ratio >= self.threshold_truncate


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


class ContextStrategy(Enum):
    """上下文管理策略。

    DEFAULT: 全量消息，无截断（适合短会话）
    SLIDING_WINDOW: 摘要旧轮次 + 保留最近 N 轮完整消息
    AGGRESSIVE: 仅保留最后一轮完整对话 + 其余历史摘要
    """

    DEFAULT = "default"
    SLIDING_WINDOW = "sliding"
    AGGRESSIVE = "aggressive"


@dataclass
class AgentContext:
    """Agent 运行时上下文，包含构建最终 prompt 的所有信息。

    Attributes:
        identity_prompt: 核心身份定义（L1）
        project_instructions: 项目指令（L2）
        tool_definitions: 可用工具定义（L6）
        session_messages: 当前会话消息列表（最终发送给 LLM 的 messages）
        session_memory: 会话级 KV 记忆（仅当前 session 有效）
        persistent_memories: 从文件系统加载的持久记忆（L4）
        environment_info: 动态环境信息（L3）
        token_budget: Token 预算与使用统计
    """

    identity_prompt: str
    project_instructions: str
    tool_definitions: list[ToolDefinition]
    session_messages: list[Message]
    session_memory: dict[str, str] = field(default_factory=dict)
    persistent_memories: list[MemoryItem] = field(default_factory=list)
    environment_info: str | None = None
    token_budget: TokenBudget = field(default_factory=TokenBudget)
