"""上下文构建器。

负责分层构建系统提示，组装最终发送给 LLM 的消息列表。
"""

from __future__ import annotations

from typing import Any

from kocor.config import config_get
from kocor.context.env_info import build_environment_info
from kocor.context.budget import TokenBudget
from kocor.context.types import AgentContext
from kocor.context.strategies import ContextStrategyApplier
from kocor.context.types import ContextStrategy
from kocor.context.project_instructions import load_project_instructions
from kocor.context.summarizer import HistorySummarizer
from kocor.context.token_counter import TokenCounter
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class ContextBuilder:
    """上下文构建器。

    负责：
    - 分层组装系统提示（L1 身份 + L2 项目指令 + L3 环境信息 + L4 记忆）
    - 构建最终消息列表（system prompt + 历史 + 当前输入）
    - Token 预算估算

    Attributes:
        identity_prompt: 核心身份定义（L1）
        tools: 工具注册表（用于获取工具定义 L6）
        memory: 记忆管理器（L4），可为 None
    """

    def __init__(
        self,
        identity_prompt: str,
        tools: Any,  # 鸭式类型：需要 get_definitions() 方法
        memory: Any | None = None,
        summarizer: HistorySummarizer | None = None,
    ):
        self.identity_prompt = identity_prompt
        self.tools = tools
        self.memory = memory
        self._token_counter = TokenCounter()
        self.summarizer = summarizer
        self.strategy_applier = (
            ContextStrategyApplier(summarizer=summarizer)
            if summarizer
            else None
        )

    def build_system_prompt(self) -> str:
        """构建完整的系统提示文本。

        组装所有可用层（L1 身份 + L2 项目指令 + L3 环境 + L4 记忆），
        用分隔线分开。

        Returns:
            多层合并后的系统提示字符串
        """
        layers = []

        # L1: 身份提示
        layers.append(self.identity_prompt)

        # L2: 项目指令
        project_instructions = load_project_instructions()
        if project_instructions:
            layers.append(project_instructions)

        # L3: 动态环境信息
        layers.append(build_environment_info())

        # L4: 持久记忆（如有）
        memories_text = self._build_memories_block()
        if memories_text:
            layers.append(memories_text)

        return "\n\n---\n\n".join(layers)

    def build_context(
        self,
        user_input: str,
        session_history: list[Message],
        strategy: ContextStrategy = ContextStrategy.DEFAULT,
    ) -> AgentContext:
        """构建完整上下文。

        Args:
            user_input: 当前用户输入
            session_history: 当前会话历史消息
            strategy: 上下文管理策略

        Returns:
            AgentContext: 包含系统提示、消息列表、预算信息的上下文对象
        """
        # 1. 构建系统提示（含 L1-L4 所有层）
        system_content = self.build_system_prompt()

        # 2. 估算总 token 用量并构建预算（用于驱动策略决策）
        estimated_total = (
            self._token_counter.count(system_content)
            + self._token_counter.count(user_input)
            + self._token_counter.count_messages(session_history)
        )
        token_budget = TokenBudget(
            limit=config_get("context_max_tokens"),
            threshold_summary=config_get("context_summary_threshold"),
            threshold_truncate=config_get("context_truncate_threshold"),
        )
        token_budget.used_prompt = estimated_total

        # 3. 应用上下文策略处理会话历史
        processed_history = session_history
        if self.strategy_applier:
            processed_history, _ = self.strategy_applier.apply(
                messages=session_history,
                strategy=strategy,
                token_budget=token_budget,
            )

        # 4. 构建消息列表（摘要已由 strategy 嵌入 processed_history）
        messages: list[Message] = [
            Message(role="system", content=system_content),
        ]
        messages.extend(processed_history)
        messages.append(Message(role="user", content=user_input))

        # 5. 计算最终 Token 预算
        token_budget.used_prompt = self._token_counter.count_messages(messages)

        # 6. 获取工具定义
        tool_definitions = self.tools.get_definitions()

        return AgentContext(
            system_content=system_content,
            tool_definitions=tool_definitions,
            session_messages=messages,
            token_budget=token_budget,
        )

    def _build_memories_block(self, max_items: int = 20) -> str:
        """构建持久记忆文本块。

        Args:
            max_items: 最多包含的记忆条数

        Returns:
            格式化的记忆文本，无记忆时返回空字符串
        """
        if not self.memory:
            return ""

        items = self.memory.list()[:max_items]
        if not items:
            return ""

        lines = ["## 已记录的信息\n"]
        for item in items:
            lines.append(f"### {item.name}")
            lines.append(item.content)
            lines.append("")

        return "\n".join(lines)