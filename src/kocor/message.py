"""消息数据模型。

内部统一格式，不依赖任何 LLM provider。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """单条消息。

    Attributes:
        role: 消息角色 (system / user / assistant / tool)
        content: 消息内容
        tool_call_id: tool 消息关联的工具调用 ID
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


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
