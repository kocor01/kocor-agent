"""上下文压缩逻辑——策略选择 + todo 注入。

与 RuntimeContext 解耦，接收数据、返回结果，不持有状态。
可根据测试需要独立实例化，无需构造整个 ContextManager。
"""

from __future__ import annotations

from typing import Any

from kocor.config import Config
from kocor.context.runtime_context import RuntimeContext
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.token_counter import TokenCounter
from kocor.context.types import ContextStrategy
from kocor.llm_provider.message import Message


class ContextCompressor:
    """上下文压缩器。

    职责：
    - 检测上下文大小是否达到压缩阈值
    - 选择合适的策略并执行压缩
    - 压缩后注入 active todo 快照

    用法:
        compressor = ContextCompressor()
        compressor.compress_if_needed(ctx=ctx, todo_store=store, ...)
    """

    def __init__(self, context_strategy: ContextStrategy | None = None):
        """初始化压缩器。

        Args:
            context_strategy: 上下文管理策略。为 None 时从 Config 读取。
        """
        self._token_counter = TokenCounter()
        self._strategy_applier = ContextStrategyApplier()
        if context_strategy is None:
            resolved = Config.load().context_strategy
            mapping = {
                "default": ContextStrategy.DEFAULT,
                "sliding": ContextStrategy.SLIDING_WINDOW,
                "aggressive": ContextStrategy.AGGRESSIVE,
            }
            self._context_strategy = mapping.get(resolved.lower(), ContextStrategy.DEFAULT)
        else:
            self._context_strategy = context_strategy

    def compress_if_needed(
        self,
        ctx: RuntimeContext,
        todo_store: Any | None,
        total_token: int | None = None,
    ) -> None:
        """检测上下文大小，必要时压缩。

        使用 API 返回的真实 token 数优先估算，
        无真实数据时回退本地 TokenCounter 估算。

        Args:
            ctx: 运行时数据（messages 会被直接修改）
            todo_store: 用于压缩后注入 active todo 快照
            total_token: 外部计算的 token 总数（如由 ContextManager 提供）。
                为 None 时 compressor 自行估算。
        """
        budget = ctx.token_budget

        if total_token is None:
            # 优先使用 API 精确计数，无时回退本地估算
            total_token = (
                (ctx.usage.prompt_tokens + ctx.usage.completion_tokens) if ctx.usage
                else (
                    self._token_counter.count_messages(ctx.messages)
                    + self._token_counter.count_tools(ctx.tool_definitions)
                )
            )

        # 判断是否达到摘要阈值
        usage_ratio = total_token / budget.limit if budget.limit > 0 else 0
        if usage_ratio < budget.threshold_summary:
            return

        system = [m for m in ctx.messages if m.role == "system"]
        history = [m for m in ctx.messages if m.role != "system"]

        processed, summary_node = self._strategy_applier.apply(
            messages=history,
            strategy=self._context_strategy,
            token_budget=budget,
        )

        ctx.messages = system + processed
        # 压缩发生时，在末尾注入 active todos 快照
        if summary_node is not None:
            self._inject_todo_snapshot(ctx, todo_store)

    @staticmethod
    def _inject_todo_snapshot(ctx: RuntimeContext, todo_store: Any | None) -> None:
        """把 active todos 作为 user 消息追加以提示 LLM。

        active 项为空时不注入，避免冗余。
        """
        if not todo_store:
            return
        snapshot = todo_store.format_for_injection()
        if snapshot is None:
            return
        ctx.messages.append(Message(role="user", content=snapshot))