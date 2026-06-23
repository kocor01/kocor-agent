"""消息数据模型。

内部统一格式，不依赖任何 LLM provider。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Usage:
    """LLM API token 用量。"""

    input: int = 0
    output: int = 0


@dataclass
class Message:
    """单条消息。

    Attributes:
        role: 消息角色 (system / user / assistant / tool)
        content: 消息内容
        tool_call_id: tool 消息关联的工具调用 ID
        reasoning: 思维链内容（用于支持推理模型的 assistant 消息）
        tool_calls: 工具调用列表
        usage: LLM API token 用量（仅在 assistant 消息中填充）
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


@dataclass
class FunctionCall:
    """工具调用的函数信息。

    Attributes:
        name: 工具名称
        arguments: JSON 字符串形式的参数
    """

    name: str
    arguments: str


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求。

    Attributes:
        id: provider 生成的调用 ID
        type: 调用类型，目前只支持 "function"
        function: 函数调用信息
    """

    id: str
    function: FunctionCall
    type: str = "function"


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        tool_call_id: 关联的工具调用 ID
        content: 执行结果内容
    """

    tool_call_id: str
    content: str


@dataclass
class StreamChunk:
    """流式输出数据块。

    Attributes:
        content: 本轮增量文本（直接 append，非累积）
        reasoning: 本轮增量思维链内容（用于支持推理模型）
        tool_calls: 本轮新增的工具调用列表
        tool_result: 工具执行结果（Agent 内部使用）
        is_final: 是否为最后一个 chunk（本次 LLM 响应结束）
        usage: LLM API token 用量（is_final=True 时填充）
    """

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None
    is_final: bool = False
    usage: Usage | None = None
