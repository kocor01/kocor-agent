"""ReAct 循环引擎。

管理 LLM 生成 → 工具执行 → 循环 的完整流程。

职责边界：Loop 是 Agent 的内部编排引擎，所需 harness 组件（权限、钩子、
事件、预算、工具）由 Agent 注入并持有。Loop 非独立可复用——它假定这些
组件已由 Agent 装配完成。调用方仅通过 run/stream（含上下文构建）或
run_messages/stream_messages（在已预设消息上直接运行循环）驱动循环。
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import Event, EventEmitter, EventType
from kocor.hook.base import HookAction, HookContext, HookPoint
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message, StreamChunk
from kocor.tools.permission import PermissionManager
from kocor.tools.tool_manager import ToolManager


class Loop:
    """ReAct 循环编排引擎（Agent 的内部组件）。

    职责：
    - 管理 ReAct 循环（LLM 生成 → 工具执行 → 循环）
    - 权限检查、钩子调用、预算追踪、事件分发
    - 循环结束后提取 session_history（状态归属在本引擎内收敛）
    """

    def __init__(
        self,
        llm: LLMClient,
        ctx: ContextManager,
        tool_manager: ToolManager,
        permission_mgr: PermissionManager,
        hook_manager: HookManager,
        event_emitter: EventEmitter,
        max_iterations: int,
    ):
        self.llm = llm
        self.ctx = ctx
        self.tool_manager = tool_manager
        self.permission_mgr = permission_mgr
        self.hook_manager = hook_manager
        self.event_emitter = event_emitter
        self.max_iterations = max_iterations

        # 重复工具调用检测
        self._consecutive_duplicate_count = 0
        self._last_tool_call_signature: str | None = None

        # 停止标志
        self._stop_requested = False

    # ── 公开方法 ──

    def run(self, user_input: str) -> str:
        """执行一次完整的 ReAct 循环。"""
        self._reset_state()
        self.ctx.build_initial_context(user_input)
        return self.run_messages()

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 ReAct 循环。"""
        self._reset_state()
        self.ctx.build_initial_context(user_input)
        yield from self.stream_messages()

    # ── 核心循环 ──

    def stop(self) -> None:
        """请求在当前迭代边界停止 ReAct 循环。"""
        self._stop_requested = True

    def _reset_state(self) -> None:
        self.ctx.reset()
        self._consecutive_duplicate_count = 0
        self._last_tool_call_signature = None
        self._stop_requested = False

    def run_messages(self) -> str:
        """运行 ReAct 循环（消息已由 build_initial_context 或调用方预设）。

        循环结束后由本方法负责将本轮 messages 提取为 session_history，
        调用方无需手工调用 extract_session_history。
        """
        try:
            while not self.ctx.iteration >= self.max_iterations:
                if self._stop_requested:
                    return self._stopped_message()

                self.ctx.advance_iteration()

                hook_msg = self._run_hooks(HookPoint.PRE_GENERATE)
                self._emit_event(EventType.PRE_GENERATE, iteration=self.ctx.iteration,
                           messages=self.ctx.messages,
                           tools=self.tool_manager.get_definitions(),
                           hook_result=hook_msg)
                if hook_msg is not None:
                    return hook_msg

                response = self.llm.generate(
                    self.ctx.messages,
                    tools=self.tool_manager.get_definitions(),
                )
                self.ctx.append(response)

                hook_msg = self._run_hooks(HookPoint.POST_GENERATE, response=response)
                self._emit_event(EventType.POST_GENERATE, iteration=self.ctx.iteration, response=response,
                           hook_result=hook_msg)
                if hook_msg is not None:
                    return hook_msg or response.content or ""

                if not response.tool_calls:
                    return response.content or ""

                if self._check_repetition(response):
                    return self._stuck_in_loop_message()

                for tool_call in response.tool_calls:
                    result_msg = self._execute_one_tool(tool_call)
                    if result_msg is not None:
                        self.ctx.append(result_msg)

                # 工具结果已追加，压缩上下文供下一轮迭代使用
                self.ctx.usage = response.usage
                self.ctx.compress_if_needed()

            hook_msg = self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
            self._emit_event(EventType.ON_BUDGET_EXHAUSTED, iteration=self.ctx.iteration,
                       max_iterations=self.max_iterations,
                       hook_result=hook_msg)
            return self._budget_exhausted_message()
        except KeyboardInterrupt:
            return self._stopped_message()
        finally:
            # 状态归属收敛：循环结束后统一提取 session_history，
            # 避免调用方（Agent）手工补位导致遗漏或不一致
            self.ctx.extract_session_history()

    def stream_messages(self) -> Iterator[StreamChunk]:
        """以流模式运行 ReAct 循环（消息已由 build_initial_context 或调用方预设）。

        生成器结束（耗尽或被关闭）时由本方法负责提取 session_history。
        """
        try:
            while not self.ctx.iteration >= self.max_iterations:
                if self._stop_requested:
                    msg = self._stopped_message()
                    yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
                    return

                self.ctx.advance_iteration()

                hook_msg = self._run_hooks(HookPoint.PRE_GENERATE)
                self._emit_event(EventType.PRE_GENERATE, iteration=self.ctx.iteration,
                           messages=self.ctx.messages,
                           tools=self.tool_manager.get_definitions(),
                           hook_result=hook_msg)
                if hook_msg is not None:
                    yield StreamChunk(content=hook_msg, is_final=True)
                    return

                accumulated_tool_calls = []
                final_content = ""
                final_reasoning = ""
                streaming_usage = None

                for chunk in self.llm.stream(
                    self.ctx.messages,
                    tools=self.tool_manager.get_definitions(),
                ):
                    # 在 LLM 流式块之间检查停止信号。
                    # 即使 KeyboardInterrupt 被延迟传递（Windows  blocked I/O），
                    # 一旦 read timeout 突破阻塞后也能在此处迅速响应。
                    if self._stop_requested:
                        msg = self._stopped_message()
                        yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
                        return

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
                    # 吸收 LLM 流的纯结束标记（is_final 且无实质内容，可能携带 usage）：
                    # 不透传给渲染层，由循环层在轮末统一发出 is_final 关闭轮次。
                    # 否则该标记会在 POST_GENERATE 钩子前提前关闭当前轮，使随后的
                    # abort/stop 消息被渲染层误判为新的一轮"第 N 次请求"。
                    if chunk.is_final and not chunk.content and not chunk.reasoning and not chunk.tool_calls:
                        continue
                    yield chunk

                response = Message(
                    role="assistant",
                    content=final_content,
                    reasoning=final_reasoning,
                    tool_calls=accumulated_tool_calls or None,
                    usage=streaming_usage,
                )

                hook_msg = self._run_hooks(HookPoint.POST_GENERATE, response=response)
                self._emit_event(EventType.POST_GENERATE, iteration=self.ctx.iteration,
                           response=response, hook_result=hook_msg)
                if hook_msg is not None:
                    yield StreamChunk(content=hook_msg or response.content or "", is_final=True)
                    return

                self.ctx.append(response)

                if not accumulated_tool_calls:
                    # 纯文本回复：LLM 流的结束标记已被吸收，由循环层补发关闭当前轮
                    yield StreamChunk(is_final=True)
                    return

                if self._check_repetition(response):
                    yield StreamChunk(content=self._stuck_in_loop_message(), is_final=True)
                    return

                for tool_call in accumulated_tool_calls:
                    if self._stop_requested:
                        msg = self._stopped_message()
                        yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
                        return

                    result_msg = self._execute_one_tool(tool_call)
                    if result_msg is not None:
                        self.ctx.append(result_msg)
                        yield StreamChunk(
                            tool_result=result_msg,
                            is_final=False,
                        )

                # 本轮工具执行完毕：循环层主动发出结束标记关闭当前渲染轮，
                # 使下一轮 LLM 生成开启新的"第 N 次请求"标题。
                # LLM 流自带的结束标记已被上方吸收，轮次边界由此处统一管控。
                yield StreamChunk(is_final=True)

                # 工具结果已追加，压缩上下文供下一轮迭代使用
                self.ctx.usage = streaming_usage
                self.ctx.compress_if_needed()

            hook_msg = self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
            self._emit_event(EventType.ON_BUDGET_EXHAUSTED, iteration=self.ctx.iteration,
                       max_iterations=self.max_iterations,
                       hook_result=hook_msg)
            yield StreamChunk(content=self._budget_exhausted_message(), is_final=True)
        except KeyboardInterrupt:
            msg = self._stopped_message()
            yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
            return
        finally:
            # 状态归属收敛：生成器结束时统一提取 session_history
            self.ctx.extract_session_history()

    # ── 工具执行 ──

    def _execute_one_tool(self, tool_call) -> Message | None:
        """执行单个工具调用：权限检查、钩子、事件、执行、审计。"""
        tool_name = tool_call.function.name

        if not self.permission_mgr.check(tool_call):
            return Message(
                role="tool",
                content="[Permission Denied] 用户拒绝了此工具调用，请勿再尝试使用此工具。",
                tool_call_id=tool_call.id,
            )

        # 先执行钩子，再触发事件（事件携带钩子结果供观察者使用）
        hook_msg = self._run_hooks(HookPoint.PRE_TOOL, tool_call=tool_call)
        self._emit_event(EventType.PRE_TOOL, iteration=self.ctx.iteration, tool_call=tool_call,
                   hook_result=hook_msg)
        if hook_msg is not None:
            # 钩子跳过工具：不触发 POST_TOOL（工具未执行），
            # 跳过事实由 PRE_TOOL 的 hook_result 表达
            return Message(role="tool", content=hook_msg or "[Tool Skipped by Hook]", tool_call_id=tool_call.id)

        duration = 0
        start = time.monotonic()
        try:
            result = self.tool_manager.execute(tool_call)
            duration = (time.monotonic() - start) * 1000
            content = result.content or ""

            hook_msg = self._run_hooks(HookPoint.POST_TOOL, tool_call=tool_call, tool_result=result)
            self._emit_event(EventType.POST_TOOL, iteration=self.ctx.iteration,
                       tool_name=tool_name, duration=duration, success=True, result=result,
                       hook_result=hook_msg)
            if hook_msg is not None:
                self._stop_requested = True

            return Message(
                role="tool",
                content=content,
                tool_call_id=getattr(result, "tool_call_id", tool_call.id),
            )

        except Exception as e:
            hook_msg = self._run_hooks(HookPoint.ON_ERROR, error=e)
            self._emit_event(EventType.POST_TOOL, iteration=self.ctx.iteration,
                       tool_name=tool_name, duration=duration, success=False, error=str(e))
            self._emit_event(EventType.ON_ERROR, iteration=self.ctx.iteration,
                       component="tool", error=str(e), hook_result=hook_msg)
            if hook_msg is not None:
                self._stop_requested = True

            return Message(
                role="tool",
                content=f"Error: {type(e).__name__}: {e}",
                tool_call_id=tool_call.id,
            )

    # ── 重复工具调用检测 ──

    @staticmethod
    def _get_tool_call_signature(tool_call) -> str:
        """将工具调用规范化为可比较的字符串签名。"""
        name = tool_call.function.name
        args = tool_call.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        return f"{name}({json.dumps(args, sort_keys=True, ensure_ascii=False)})"

    def _check_repetition(self, response: Message) -> bool:
        """检测 LLM 是否在重复调用相同的工具。

        检测到第 2 次重复时注入显式警告让模型自行纠正；
        第 3 次重复时触发 ON_BUDGET_EXHAUSTED 事件和钩子并返回 True。
        """
        if not response.tool_calls:
            self._consecutive_duplicate_count = 0
            self._last_tool_call_signature = None
            return False

        signatures = [self._get_tool_call_signature(tc) for tc in response.tool_calls]
        combined = "|".join(signatures)

        if combined == self._last_tool_call_signature:
            self._consecutive_duplicate_count += 1
        else:
            self._consecutive_duplicate_count = 1
            self._last_tool_call_signature = combined

        # 第 2 次重复时注入显式警告，让模型在下一轮看到并停止
        if self._consecutive_duplicate_count == 2:
            self.ctx.append(Message(
                role="user",
                content="[SYSTEM] 检测到重复的工具调用。你正在重复调用相同的工具，请立即停止并回复用户。",
            ))

        if self._consecutive_duplicate_count >= 3:
            hook_msg = self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
            self._emit_event(EventType.ON_BUDGET_EXHAUSTED, iteration=self.ctx.iteration,
                       max_iterations=self.max_iterations, reason="duplicate_tool_calls",
                       hook_result=hook_msg)
            return True

        return False

    # ── 辅助方法 ──

    def _budget_exhausted_message(self) -> str:
        return f"Agent 在 {self.ctx.iteration} 次迭代后未完成。"

    def _stuck_in_loop_message(self) -> str:
        return (
            f"Agent 在第 {self.ctx.iteration} 次迭代检测到重复工具调用"
            f"（连续 {self._consecutive_duplicate_count} 次），已提前终止。"
        )

    def _stopped_message(self) -> str:
        self._stop_requested = False
        return f"Agent 在第 {self.ctx.iteration} 次迭代被终止。"

    def _run_hooks(self, point: HookPoint, **extra) -> str | None:
        """执行指定生命周期点的钩子。

        返回 abort 消息（钩子终止循环）或 None（继续执行）。

        已知字段传给 HookContext 命名参数，未知字段放入 extra 字典，
        避免因未知字段导致 TypeError（P0.2 回归）。
        """
        known_fields = {"tool_call", "tool_result", "response", "error", "config"}
        context_kwargs = {}
        extra_data = {}
        for k, v in extra.items():
            if k in known_fields:
                context_kwargs[k] = v
            else:
                extra_data[k] = v
        context_kwargs["extra"] = extra_data

        ctx = HookContext(
            iteration=self.ctx.iteration,
            messages=self.ctx.messages,
            **context_kwargs,
        )
        for r in self.hook_manager.run(point, ctx):
            if r.action == HookAction.ABORT:
                return r.message
        return None

    def _emit_event(self, event_type: str, **data) -> None:
        self.event_emitter.fire(Event(
            type=event_type,
            iteration=self.ctx.iteration,
            data=data,
            timestamp=time.time(),
        ))