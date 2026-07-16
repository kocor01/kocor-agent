"""ReAct 循环的运行时数据——纯数据容器，不包含任何业务逻辑。

由 ContextManager 持有，通过属性委托暴露给 Loop 读取。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kocor.context.budget import TokenBudget
from kocor.llm_provider.message import Message, Usage
from kocor.tools.definitions import ToolDefinition


@dataclass
class RuntimeContext:
    """ReAct 循环的运行时数据。

    职责：纯数据容器，承载一轮或多轮对话中累积的消息、迭代计数、Token 预算。
    不包含任何业务逻辑——构建、压缩、提取等操作由 ContextManager 编排。

    Attributes:
        system_content: 系统提示文本
        tool_definitions: 可用工具定义
        messages: 当前完整消息列表（含 system）
        token_budget: Token 预算与使用统计
        session_history: 跨 run() 调用的会话历史
        iteration: 当前轮次迭代次数
        usage: 最近一次 LLM 返回的真实 token 用量
    """

    system_content: str = ""
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    session_history: list[Message] = field(default_factory=list)
    usage: Usage | None = None
    iteration: int = 0

    def append(self, message: Message) -> None:
        """追加一条消息到 messages。"""
        self.messages.append(message)

    def reset(self) -> None:
        """重置运行时数据（不清除 session_history）。"""
        self.iteration = 0
        self.messages.clear()
        self.token_budget.reset()
        self.usage = None

    def reset_conversation(self) -> None:
        """重置运行时数据并清除会话历史。"""
        self.reset()
        self.session_history.clear()