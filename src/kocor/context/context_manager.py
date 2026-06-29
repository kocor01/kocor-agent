"""Agent 运行时上下文管理器。

统一封装消息积累、跨轮历史、迭代追踪、上下文压缩。
既是数据承载，也负责分层组装系统提示和上下文构建。
"""

from __future__ import annotations

from typing import Any

from kocor.config import Config
from kocor.context.budget import TokenBudget
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.system_prompt import SystemPromptBuilder
from kocor.context.token_counter import TokenCounter
from kocor.context.types import ContextStrategy
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class ContextManager:
    """Agent 运行时上下文管理器。

    核心依赖通过构造函数传入，其余依赖按需取用（Config 单例、延迟创建）。

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
        tools: Any | None = None,  # 鸭式类型：需要 get_definitions() 方法
        memory: Any | None = None,
    ):
        # 核心依赖
        self.tools = tools
        self.memory = memory

        # Token 计数
        self._token_counter = TokenCounter()

        # 系统提示构建器
        self._prompt_builder = SystemPromptBuilder(memory)

        # 上下文策略
        resolved = Config.get("context_strategy")
        mapping = {
            "default": ContextStrategy.DEFAULT,
            "sliding": ContextStrategy.SLIDING_WINDOW,
            "aggressive": ContextStrategy.AGGRESSIVE,
        }
        self._context_strategy = mapping.get(resolved.lower(), ContextStrategy.DEFAULT)

        # 策略执行器
        self._strategy_applier = ContextStrategyApplier()

        # 数据字段（由 build_initial_context 填充）
        self.system_content = ""
        self.tool_definitions: list[ToolDefinition] = []
        self.messages: list[Message] = []
        self.token_budget = TokenBudget()

        # 跨轮状态
        self.session_history: list[Message] = []

        # 本轮状态
        self.iteration = 0

    # ── 公开方法 ──

    def build_initial_context(self, user_input: str) -> None:
        """构建本轮初始上下文：系统提示、消息列表、Token 预算。"""
        # 1. 构建系统提示（含 L1-L4 所有层）
        self.system_content = self._prompt_builder.build()

        # 2. 获取工具定义
        self.tool_definitions = self.tools.get_definitions() if self.tools else []
        tool_tokens = self._token_counter.count_tools(self.tool_definitions)

        # 3. 估算总 token 用量并构建预算
        estimated_total = (
            self._token_counter.count(self.system_content)
            + self._token_counter.count(user_input)
            + self._token_counter.count_messages(self.session_history)
            + tool_tokens
        )
        self.token_budget = TokenBudget(
            limit=Config.get("context_max_tokens"),
            threshold_summary=Config.get("context_summary_threshold"),
            threshold_truncate=Config.get("context_truncate_threshold"),
        )
        self.token_budget.used_prompt = estimated_total

        # 4. 应用上下文策略处理会话历史
        processed_history = self.session_history
        if self._strategy_applier:
            processed_history, _ = self._strategy_applier.apply(
                messages=self.session_history,
                strategy=self._context_strategy,
                token_budget=self.token_budget,
            )

        # 5. 构建消息列表
        self.messages = [
            Message(role="system", content=self.system_content),
        ]
        self.messages.extend(processed_history)
        self.messages.append(Message(role="user", content=user_input))
        self.token_budget.used_prompt = self._token_counter.count_messages(self.messages) + tool_tokens

    def count_message_tokens(self, messages: list[Message]) -> int:
        """估算消息列表的 token 数。"""
        return self._token_counter.count_messages(messages)

    def count_tool_tokens(self) -> int:
        """估算工具定义的 token 数。"""
        return self._token_counter.count_tools(self.tool_definitions)

    def compress_if_needed(self) -> None:
        """每次 LLM 请求前检测上下文大小，必要时压缩。

        保留 system 消息，对非 system 部分应用策略（滑动窗口/摘要）。
        """
        if (self._context_strategy == ContextStrategy.DEFAULT
                or not self._strategy_applier):
            return

        tool_tokens = self.count_tool_tokens()
        total = self.count_message_tokens(self.messages) + tool_tokens
        budget = TokenBudget()
        budget.used_prompt = total

        if not budget.should_summarize():
            return

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
