"""Agent 核心。

Agent 身份、slash 命令路由、组件组装。ReAct 循环由 Loop 引擎驱动。
"""

from __future__ import annotations

from typing import Iterator

from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.context.memory import MemoryManager
from kocor.harness.budget import IterationBudget
from kocor.harness.event.event_manager import EventEmitter
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message, StreamChunk
from kocor.loop import Loop
from kocor.skill.types import InvokeStrategy, SkillContext, SkillType
from kocor.tools.permission import PermissionManager
from kocor.tools.tool_manager import ToolManager


class Agent:
    """自主 Agent 核心。

    职责：
    - slash 命令识别和调度
    - Agent 身份管理（LLM、工具、上下文）
    - 委托 Loop 执行 ReAct 循环
    """

    def __init__(
        self,
        llm: LLMClient,
        tool_manager: ToolManager | None = None,
        # Harness 参数（可选）
        permission_mgr: PermissionManager | None = None,
        hook_manager: HookManager | None = None,
        event_emitter: EventEmitter | None = None,
        budget: IterationBudget | None = None,
    ):
        self.llm = llm
        self.tool_manager = tool_manager or ToolManager()
        self.system_prompt = Config.get("default_system_prompt")
        self.max_iterations = Config.get("max_iterations")

        # Harness 组件
        self.permission_mgr = permission_mgr or PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE)
        self.hook_manager = hook_manager or HookManager()
        self.event_emitter = event_emitter or EventEmitter()
        self.budget = budget or IterationBudget(iterations_limit=self.max_iterations)

        # 上下文管理
        memory: MemoryManager | None = None
        memory_dir = Config.get("memory_dir") or None
        if memory_dir:
            memory = MemoryManager(memory_dir=memory_dir)

        # 运行时上下文管理器
        self.ctx = ContextManager(
            tools=self.tool_manager,
            memory=memory,
        )

        # ReAct 循环引擎
        self.loop = Loop(
            llm=self.llm,
            ctx=self.ctx,
            tool_manager=self.tool_manager,
            permission_mgr=self.permission_mgr,
            hook_manager=self.hook_manager,
            event_emitter=self.event_emitter,
            budget=self.budget,
        )

    # ── 公开方法 ──

    def run(self, user_input: str) -> str:
        """执行一次完整的 Agent 循环。"""
        if self.tool_manager.skill_manager and user_input.startswith("/"):
            return self._handle_slash_command(user_input)
        return self.loop.run(user_input)

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 Agent 循环。"""
        if self.tool_manager.skill_manager and user_input.startswith("/"):
            result = self._handle_slash_command(user_input)
            yield StreamChunk(content=result, is_final=True)
            return
        yield from self.loop.stream(user_input)

    def reset_conversation(self) -> None:
        """清空会话历史，开始新对话。"""
        self.ctx.reset_conversation()

    # ── slash 命令 ──

    def _handle_slash_command(self, user_input: str) -> str:
        parts = user_input[1:].strip().split(maxsplit=1)
        skill_name = parts[0]
        skill_args = parts[1] if len(parts) > 1 else ""

        skill_def = self.tool_manager.skill_manager.get(skill_name)
        if skill_def is None:
            available = self._list_slash_skills()
            return f"Unknown skill: '{skill_name}'. Available: {available}"

        if skill_def.invoke_strategy not in (InvokeStrategy.SLASH, InvokeStrategy.BOTH):
            return f"Skill '{skill_name}' cannot be invoked via slash command."

        context = SkillContext(
            user_input=skill_args,
            tool_manager=self.tool_manager,
        )
        result = self.tool_manager.skill_manager.execute(skill_name, context)

        if not result.success:
            return result.content

        if skill_def.skill_type == SkillType.PROMPT:
            messages = [
                Message(role="system", content=self.system_prompt),
            ]
            if skill_def.prompt_role == "system":
                messages.append(Message(role="system", content=result.content))
            else:
                messages.append(Message(role="user", content=result.content))
            self.ctx.reset()
            self.ctx.messages = messages
            return self.loop._run_messages()
        else:
            return result.content

    def _list_slash_skills(self) -> str:
        names = [
            f"/{s.name}"
            for s in self.tool_manager.skill_manager.list_skills(enabled_only=True)
            if s.invoke_strategy in (InvokeStrategy.SLASH, InvokeStrategy.BOTH)
        ]
        return ", ".join(sorted(names))
