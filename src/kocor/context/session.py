"""Agent 运行时上下文管理器。

统一封装消息积累、跨轮历史、迭代追踪、工具记录、上下文压缩。
既是 ContextBuilder.build_context() 的返回类型，也是 Agent 持有的管理器。
"""

from __future__ import annotations

from kocor.context.budget import TokenBudget
from kocor.context.builder import ContextBuilder
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.types import ContextStrategy
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class AgentContext:
    """Agent 运行时上下文管理器。

    既是 ContextBuilder.build_context() 的返回类型（数据承载），
    也负责消息积累、跨轮历史、迭代追踪、工具记录和上下文压缩（状态管理）。

    Attributes:
        system_content: 系统提示文本
        tool_definitions: 可用工具定义
        messages: 当前完整消息列表（含 system）
        token_budget: Token 预算与使用统计
        session_history: 跨 run() 调用的会话历史
        iteration: 当前轮次迭代次数
    """

    def __init__(
        self,
        system_content: str = "",
        tool_definitions: list[ToolDefinition] | None = None,
        messages: list[Message] | None = None,
        token_budget: TokenBudget | None = None,
        # 管理依赖（Agent 设置）
        context_builder: ContextBuilder | None = None,
        context_strategy: ContextStrategy = ContextStrategy.DEFAULT,
        strategy_applier: ContextStrategyApplier | None = None,
    ):
        # 数据字段（build_context() 填充）
        self.system_content = system_content
        self.tool_definitions = tool_definitions or []
        self.messages = messages or []
        self.token_budget = token_budget or TokenBudget()

        # 跨轮状态
        self.session_history: list[Message] = []

        # 本轮状态
        self.iteration = 0

        # 管理依赖
        self._context_builder = context_builder
        self._context_strategy = context_strategy
        self._strategy_applier = strategy_applier

    def build_initial_context(self, user_input: str) -> None:
        """调用 ContextBuilder 构建本轮初始消息。

        将 session_history 传入 builder，获取包含历史处理的初始消息列表。
        """
        data = self._context_builder.build_context(
            user_input=user_input,
            session_history=self.session_history,
        )
        self.system_content = data.system_content
        self.tool_definitions = data.tool_definitions
        self.messages = data.messages
        self.token_budget = data.token_budget

    def compress_if_needed(self) -> None:
        """每次 LLM 请求前检测上下文大小，必要时压缩。

        保留 system 消息，对非 system 部分应用策略（滑动窗口/摘要）。
        """
        if (self._context_strategy == ContextStrategy.DEFAULT
                or not self._strategy_applier):
            return

        # 计算当前 token 用量
        tool_tokens = self._context_builder.count_tool_tokens()
        total = self._context_builder.count_message_tokens(self.messages) + tool_tokens
        budget = TokenBudget()
        budget.used_prompt = total

        if not budget.should_summarize():
            return

        # 分离 system 和可压缩部分
        system = [m for m in self.messages if m.role == "system"]
        history = [m for m in self.messages if m.role != "system"]

        processed, _ = self._strategy_applier.apply(
            messages=history,
            strategy=self._context_strategy,
            token_budget=budget,
        )

        self.messages = system + processed

    def extract_session_history(self) -> None:
        """从本轮 messages 提取非 system 消息作为跨轮历史。"""
        self.session_history = [
            m for m in self.messages if m.role != "system"
        ]

    def advance_iteration(self) -> None:
        """迭代计数+1。"""
        self.iteration += 1

    def append(self, message: Message) -> None:
        """追加消息到当前列表。"""
        self.messages.append(message)

    def reset(self) -> None:
        """重置本轮状态（保留 session_history 用于跨轮）。"""
        self.iteration = 0
        self.messages.clear()
        self.token_budget.reset()

    def reset_conversation(self) -> None:
        """重置所有状态（含跨轮历史）。"""
        self.reset()
        self.session_history.clear()
