"""Agent 循环的审计记录类型。"""

from dataclasses import dataclass


@dataclass
class ToolCallRecord:
    """单个工具调用的不可变审计记录。"""

    iteration: int
    tool_name: str
    arguments: dict
    result_summary: str
    result_token_count: int
    duration_ms: float
    permission: str  # "auto" | "confirm" | "denied"
    error: str | None = None
