"""上下文构建器。

负责分层构建系统提示，组装最终发送给 LLM 的消息列表。
"""

from __future__ import annotations

from typing import Any

from kocor.context.env_info import build_environment_info
from kocor.context.models import AgentContext, ContextStrategy, TokenBudget
from kocor.context.project_instructions import load_project_instructions
from kocor.context.strategies import apply_context_strategy
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
        project_instructions_path: 项目指令文件路径（L2）
        max_tokens: 上下文窗口上限
    """

    def __init__(
        self,
        identity_prompt: str,
        tools: Any,  # 鸭式类型：需要 get_definitions() 方法
        memory: Any | None = None,
        project_instructions_path: str = "KOCOR.md",
        max_tokens: int = 200_000,
        summarizer: HistorySummarizer | None = None,
    ):
        self.identity_prompt = identity_prompt
        self.tools = tools
        self.memory = memory
        self.project_instructions_path = project_instructions_path
        self.max_tokens = max_tokens
        self._token_counter = TokenCounter()
        self.summarizer = summarizer

    def build_system_prompt(self, project_instructions: str | None = None) -> str:
        """构建完整的系统提示文本。

        组装所有可用层，用分隔线分开。

        Args:
            project_instructions: 已加载的项目指令文本，None 则内部加载

        Returns:
            多层合并后的系统提示字符串
        """
        layers = []

        # L1: 身份提示
        layers.append(self.identity_prompt)

        # L2: 项目指令
        if project_instructions is None:
            project_instructions = load_project_instructions(self.project_instructions_path)
        if project_instructions:
            layers.append(project_instructions)

        # L3: 动态环境信息
        env_info = build_environment_info()
        layers.append(f"## 环境信息\n{env_info}")

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
            strategy: 上下文管理策略（应用于会话历史）

        Returns:
            AgentContext: 包含系统提示、消息列表、预算信息的上下文对象
        """
        # 1. 加载项目指令（仅一次，供 L2 和 AgentContext 共用）
        project_instructions = load_project_instructions(self.project_instructions_path)

        # 2. 构建系统提示（含 L1-L4 所有层）
        system_content = self.build_system_prompt(project_instructions)

        # 3. 应用上下文策略处理会话历史
        summary_node = None
        processed_history = session_history
        if strategy != ContextStrategy.DEFAULT and self.summarizer:
            from kocor.context.models import TokenBudget as _TB
            budget = _TB(limit=self.max_tokens)
            budget.used_prompt = self._token_counter.count(system_content) + self._token_counter.count(user_input)
            processed_history, summary_node = apply_context_strategy(
                messages=session_history,
                token_budget=budget,
                summarizer=self.summarizer,
                strategy=strategy,
            )

        # 4. 构建消息列表
        messages: list[Message] = [
            Message(role="system", content=system_content),
        ]
        if summary_node:
            messages.append(Message(
                role="system",
                content=f"[历史对话摘要]\n{summary_node.summary}",
            ))
        messages.extend(processed_history)
        messages.append(Message(role="user", content=user_input))

        # 5. 计算 Token 预算
        token_budget = TokenBudget(limit=self.max_tokens)
        token_budget.used_prompt = self._token_counter.count_messages(messages)

        # 4. 获取工具定义
        tool_definitions = self.tools.get_definitions()

        return AgentContext(
            identity_prompt=self.identity_prompt,
            project_instructions=project_instructions,
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