"""Agent 核心。

单一 ReAct 循环引擎：query LLM → call tool → observe → loop until final answer。
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from kocor.config import Config
from kocor.context.memory import MemoryManager
from kocor.context.context_manager import ContextManager
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message, StreamChunk
from kocor.skill.types import InvokeStrategy, SkillContext, SkillType
from kocor.tools.tool_manager import ToolManager

from kocor.harness.budget import IterationBudget
from kocor.harness.event.event_manager import HarnessEvent, EventEmitter, EventType
from kocor.tools.permission import PermissionManager
from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction
from kocor.hook.hook_manager import HookManager


class Agent:
    """自主 Agent 核心 — 唯一的 ReAct 循环引擎。

    职责：
    - slash 命令识别和调度
    - 管理 ReAct 循环（LLM 生成 → 工具执行 → 循环）
    - 权限检查、钩子调用、预算追踪、事件分发、日志记录
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

    # ── 公开方法 ──

    def run(self, user_input: str) -> str:
        """执行一次完整的 Agent 循环。"""
        if self.tool_manager.skill_manager and user_input.startswith("/"):
            return self._handle_slash_command(user_input)

        self.ctx.reset()
        self.ctx.build_initial_context(user_input)
        result = self._run_messages()
        self.ctx.extract_session_history()
        return result

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 Agent 循环。"""
        if self.tool_manager.skill_manager and user_input.startswith("/"):
            result = self._handle_slash_command(user_input)
            yield StreamChunk(content=result, is_final=True)
            return

        self.ctx.reset()
        self.ctx.build_initial_context(user_input)
        yield from self._stream_messages()
        self.ctx.extract_session_history()

    def reset_conversation(self) -> None:
        """清空会话历史，开始新对话。"""
        self.ctx.reset_conversation()

    # ── 核心循环 ──

    def _reset_state(self) -> None:
        self.ctx.reset()
        self.budget.reset()

    def _run_messages(self) -> str:
        """运行 ReAct 循环（消息已由 build_initial_context 设置）。"""

        while not self.budget.exhausted:
            self.ctx.advance_iteration()
            self.budget.iterations_used = self.ctx.iteration

            self.ctx.compress_if_needed()

            self._emit(EventType.PRE_GENERATE, iteration=self.ctx.iteration, messages=self.ctx.messages,
                       tools=self.tool_manager.get_definitions())
            self._run_hooks(HookPoint.PRE_GENERATE)

            response = self.llm.generate(
                self.ctx.messages,
                tools=self.tool_manager.get_definitions(),
            )
            self.ctx.append(response)

            self._emit(EventType.POST_GENERATE, iteration=self.ctx.iteration, response=response)
            self._run_hooks(HookPoint.POST_GENERATE)

            if not response.tool_calls:
                return response.content or ""

            for tool_call in response.tool_calls:
                result_msg = self._execute_one_tool(tool_call)
                if result_msg is not None:
                    self.ctx.append(result_msg)

        self._emit(EventType.ON_BUDGET_EXHAUSTED, iteration=self.ctx.iteration,
                       iterations_limit=self.budget.iterations_limit)
        self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
        return self._budget_exhausted_message()

    def _stream_messages(self) -> Iterator[StreamChunk]:
        """以流模式运行 ReAct 循环（消息已由 build_initial_context 设置）。"""

        while not self.budget.exhausted:
            self.ctx.advance_iteration()
            self.budget.iterations_used = self.ctx.iteration

            self.ctx.compress_if_needed()

            self._emit(EventType.PRE_GENERATE, iteration=self.ctx.iteration, messages=self.ctx.messages,
                       tools=self.tool_manager.get_definitions())
            self._run_hooks(HookPoint.PRE_GENERATE)

            accumulated_tool_calls = []
            final_content = ""
            final_reasoning = ""
            streaming_usage = None

            for chunk in self.llm.stream(
                self.ctx.messages,
                tools=self.tool_manager.get_definitions(),
            ):
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        if not any(t.id == tc.id for t in accumulated_tool_calls):
                            accumulated_tool_calls.append(tc)
                if chunk.content:
                    final_content += chunk.content
                if chunk.reasoning:
                    final_reasoning += chunk.reasoning
                if chunk.usage:
                    streaming_usage = chunk.usage
                if chunk.usage and not chunk.content and not chunk.reasoning and not chunk.tool_calls:
                    continue
                yield chunk

            response = Message(
                role="assistant",
                content=final_content,
                reasoning=final_reasoning,
                tool_calls=accumulated_tool_calls or None,
                usage=streaming_usage,
            )

            self._emit(EventType.POST_GENERATE, iteration=self.ctx.iteration,
                       response=response)
            self._run_hooks(HookPoint.POST_GENERATE)

            self.ctx.append(response)

            if not accumulated_tool_calls:
                return

            for tool_call in accumulated_tool_calls:
                result_msg = self._execute_one_tool(tool_call)
                if result_msg is not None:
                    self.ctx.append(result_msg)
                    yield StreamChunk(
                        tool_result=result_msg,
                        is_final=False,
                    )

        self._emit(EventType.ON_BUDGET_EXHAUSTED, iteration=self.ctx.iteration,
                       iterations_limit=self.budget.iterations_limit)
        self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
        yield StreamChunk(content=self._budget_exhausted_message(), is_final=True)

    # ── 工具执行 ──

    def _execute_one_tool(self, tool_call) -> Message | None:
        """执行单个工具调用：事件、权限检查、钩子、执行、审计。"""
        tool_name = tool_call.function.name

        self._emit(EventType.PRE_TOOL, iteration=self.ctx.iteration, tool_call=tool_call)

        if not self.permission_mgr.check(tool_call):
            return Message(
                role="tool",
                content="[Permission Denied] 用户拒绝了此工具调用，请勿再尝试使用此工具。",
                tool_call_id=tool_call.id,
            )

        hook_results = self._run_hooks(HookPoint.PRE_TOOL, tool_call=tool_call)
        if any(r.action == HookAction.SKIP_TOOL for r in hook_results):
            self._emit(EventType.POST_TOOL, iteration=self.ctx.iteration,
                       tool_name=tool_name, success=False, skipped_by_hook=True)
            return Message(
                role="tool",
                content="[Tool Skipped by Hook]",
                tool_call_id=tool_call.id,
            )

        duration = 0
        start = time.monotonic()
        try:
            result = self.tool_manager.execute(tool_call)
            duration = (time.monotonic() - start) * 1000
            content = result.content or ""
            truncated = self._truncate_tool_output(content)

            self._emit(EventType.POST_TOOL, iteration=self.ctx.iteration,
                       tool_name=tool_name, duration=duration, success=True, result=result)
            self._run_hooks(HookPoint.POST_TOOL, tool_call=tool_call, tool_result=result)

            return Message(
                role="tool",
                content=truncated,
                tool_call_id=getattr(result, "tool_call_id", tool_call.id),
            )

        except Exception as e:
            self._emit(EventType.POST_TOOL, iteration=self.ctx.iteration,
                       tool_name=tool_name, duration=duration, success=False, error=str(e))
            self._emit(EventType.ON_ERROR, iteration=self.ctx.iteration,
                       component="tool", error=str(e))
            self._run_hooks(HookPoint.ON_ERROR, error=e)

            return Message(
                role="tool",
                content=f"Error: {type(e).__name__}: {e}",
                tool_call_id=tool_call.id,
            )

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
            return self._run_messages()
        else:
            return result.content

    def _list_slash_skills(self) -> str:
        names = [
            f"/{s.name}"
            for s in self.tool_manager.skill_manager.list_skills(enabled_only=True)
            if s.invoke_strategy in (InvokeStrategy.SLASH, InvokeStrategy.BOTH)
        ]
        return ", ".join(sorted(names))

    # ── 辅助方法 ──

    def _budget_exhausted_message(self) -> str:
        return f"Agent 在 {self.ctx.iteration} 次迭代后未完成。"

    @staticmethod
    def _truncate_tool_output(content: str) -> str:
        if len(content) > 50_000:
            return content[:25_000] + "\n\n...[truncated]...\n\n" + content[-25_000:]
        if len(content.splitlines()) > 2_000:
            lines = content.splitlines()
            return "\n".join(lines[:1_000] + ["...[truncated lines]..."] + lines[-1_000:])
        return content

    def _run_hooks(self, point: HookPoint, **extra) -> list[HookResult]:
        ctx = HookContext(
            iteration=self.ctx.iteration,
            messages=self.ctx.messages,
            **extra,
        )
        return self.hook_manager.run(point, ctx)

    def _emit(self, event_type: str, **data) -> None:
        self.event_emitter.fire(HarnessEvent(
            type=event_type,
            iteration=self.ctx.iteration,
            data=data,
            timestamp=time.time(),
        ))
