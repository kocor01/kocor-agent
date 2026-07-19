"""Agent 运行时上下文管理器。

编排 RuntimeContext 数据容器 + ContextCompressor 压缩逻辑 + SystemPromptBuilder 提示构建。
数据与逻辑分离：数据在 RuntimeContext 中，编排在 ContextManager 中，压缩在 ContextCompressor 中。
"""

from __future__ import annotations

from typing import Any

from kocor.config import Config
from kocor.context.budget import TokenBudget
from kocor.context.compressor import ContextCompressor
from kocor.context.runtime_context import RuntimeContext
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.system_prompt import SystemPromptBuilder
from kocor.context.token_counter import TokenCounter
from kocor.context.types import resolve_strategy
from kocor.llm_provider.message import Message, Usage
from kocor.tools.definitions import ToolDefinition


class ContextManager:
    """Agent 运行时上下文编排器。

    持有 RuntimeContext 数据实例，提供构建/压缩/提取等编排方法。
    调用方（Loop/Agent）通过属性委托访问数据，无需感知内部结构变化。

    核心依赖通过构造函数传入，其余依赖按需取用（Config 单例、延迟创建）。

    Attributes:
        ctx: RuntimeContext 实例（数据容器，通过属性委托暴露字段）
    """

    def __init__(
        self,
        tools: Any | None = None,
        memory: Any | None = None,
        todo_store: Any | None = None,
    ):
        self.tools = tools
        self.memory = memory
        self.todo_store = todo_store

        # 数据容器（纯数据，无逻辑）
        self._runtime = RuntimeContext()

        # 逻辑组件（编排器组合）
        self._token_counter = TokenCounter()
        self._prompt_builder = SystemPromptBuilder(memory)
        self._context_strategy = resolve_strategy(Config.load().context_strategy)
        self._compressor = ContextCompressor(context_strategy=self._context_strategy)
        self._strategy_applier = ContextStrategyApplier()

    # ── 属性委托（向后兼容，不改动 Loop/Agent 调用方） ──

    @property
    def system_content(self) -> str:
        return self._runtime.system_content

    @system_content.setter
    def system_content(self, value: str) -> None:
        self._runtime.system_content = value

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return self._runtime.tool_definitions

    @tool_definitions.setter
    def tool_definitions(self, value: list[ToolDefinition]) -> None:
        self._runtime.tool_definitions = value

    @property
    def messages(self) -> list[Message]:
        return self._runtime.messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._runtime.messages = value

    @property
    def token_budget(self) -> TokenBudget:
        return self._runtime.token_budget

    @token_budget.setter
    def token_budget(self, value: TokenBudget) -> None:
        self._runtime.token_budget = value

    @property
    def session_history(self) -> list[Message]:
        return self._runtime.session_history

    @session_history.setter
    def session_history(self, value: list[Message]) -> None:
        self._runtime.session_history = value

    @property
    def usage(self) -> Usage | None:
        return self._runtime.usage

    @usage.setter
    def usage(self, value: Usage | None) -> None:
        self._runtime.usage = value

    @property
    def iteration(self) -> int:
        return self._runtime.iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self._runtime.iteration = value

    # ── 公开方法 ──

    def build_initial_context(self, user_input: str) -> None:
        """构建本轮初始上下文：系统提示、消息列表、Token 预算。

        每次构建前刷新记忆快照，使 LLM 看到最新的持久化记忆。
        """
        # 每次对话构建前刷新记忆快照，使新记忆对 LLM 可见
        if self.memory:
            self.memory.refresh_snapshot()

        self._runtime.system_content = self._prompt_builder.build()

        self._runtime.tool_definitions = self.tools.get_definitions() if self.tools else []
        tool_tokens = self._token_counter.count_tools(self._runtime.tool_definitions)

        estimated_total = (
            self._token_counter.count(self._runtime.system_content)
            + self._token_counter.count(user_input)
            + self._token_counter.count_messages(self._runtime.session_history)
            + tool_tokens
        )
        self._runtime.token_budget = TokenBudget(
            limit=Config.load().context_max_tokens,
            threshold_summary=Config.load().context_summary_threshold,
            threshold_truncate=Config.load().context_truncate_threshold,
        )
        self._runtime.token_budget.used_prompt = estimated_total

        processed_history = self._runtime.session_history
        summary_node = None
        if self._strategy_applier:
            processed_history, summary_node = self._strategy_applier.apply(
                messages=self._runtime.session_history,
                strategy=self._context_strategy,
                token_budget=self._runtime.token_budget,
            )

        self._runtime.messages = [
            Message(role="system", content=self._runtime.system_content),
        ]
        self._runtime.messages.extend(processed_history)
        # 上下文压缩发生时，注入 active todos 快照，防止 LLM 重做已完成任务
        if summary_node is not None:
            self._inject_todo_snapshot()
        self._runtime.messages.append(Message(role="user", content=user_input))
        self._runtime.token_budget.used_prompt = (
            self._token_counter.count_messages(self._runtime.messages) + tool_tokens
        )

    def count_message_tokens(self) -> int:
        """计算当前消息列表的 token 数。"""
        return self._token_counter.count_messages(self._runtime.messages)

    def count_tool_tokens(self) -> int:
        """计算当前工具定义的 token 数。"""
        return self._token_counter.count_tools(self._runtime.tool_definitions)

    def compress_if_needed(self) -> None:
        """检测上下文大小，必要时压缩。

        委托给 ContextCompressor，优先使用 API 返回的真实 token 数。
        """
        total_token = (
            (self.usage.prompt_tokens + self.usage.completion_tokens) if self.usage
            else (self.count_message_tokens() + self.count_tool_tokens())
        )
        self._compressor.compress_if_needed(
            ctx=self._runtime,
            todo_store=self.todo_store,
            total_token=total_token,
        )

    def _inject_todo_snapshot(self) -> None:
        """把 active todos 作为 user 消息追加以提示 LLM。"""
        if not self.todo_store:
            return
        snapshot = self.todo_store.format_for_injection()
        if snapshot is None:
            return
        self._runtime.messages.append(Message(role="user", content=snapshot))

    def extract_session_history(self) -> None:
        """从本轮 messages 提取非 system 消息作为跨轮历史。"""
        self._runtime.session_history = [
            m for m in self._runtime.messages if m.role != "system"
        ]

    def advance_iteration(self) -> None:
        """轮次计数加 1。"""
        self._runtime.iteration += 1

    def append(self, message: Message) -> None:
        """向当前消息列表追加一条消息。"""
        self._runtime.messages.append(message)

    def reset(self) -> None:
        """重置运行时上下文（含所有状态）。"""
        self._runtime.reset()

    def reset_conversation(self) -> None:
        """重置会话（保留 system prompt 和工具定义）。"""
        self._runtime.reset_conversation()