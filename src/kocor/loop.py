"""ReAct 循环引擎。

管理 LLM 生成 → 工具执行 → 循环 的完整流程。

职责边界：Loop 是 Agent 的内部编排引擎，所需 harness 组件（权限、钩子、
事件、预算、工具）由 Agent 注入并持有。Loop 非独立可复用——它假定这些
组件已由 Agent 装配完成。调用方仅通过 run/stream（含上下文构建）或
run_messages/stream_messages（在已预设消息上直接运行循环）驱动循环。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator

logger = logging.getLogger(__name__)

from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import Event, EventEmitter, EventType
from kocor.hook.base import HookAction, HookContext, HookPoint
from kocor.hook.hook_manager import HookManager
from kocor._stream_session import StreamSession
from kocor.llm_provider.exceptions import LLMConnectionError, LLMTimeoutError
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
        context: ContextManager,
        tool_manager: ToolManager,
        permission_mgr: PermissionManager,
        hook_manager: HookManager,
        event_emitter: EventEmitter,
        max_iterations: int,
    ):
        self.llm = llm
        self.context = context
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

        # LLM 超时重试计数
        self._consecutive_timeouts = 0
        self._max_timeout_retries = 2

    # ── 公开方法 ──

    def run(self, user_input: str) -> str:
        """执行一次完整的 ReAct 循环。"""
        self._reset_state()
        self.context.build_initial_context(user_input)
        return self.run_messages()

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 ReAct 循环。"""
        self._reset_state()
        self.context.build_initial_context(user_input)
        yield from self.stream_messages()

    # ── 核心循环 ──

    def stop(self) -> None:
        """请求在当前迭代边界停止 ReAct 循环。"""
        self._stop_requested = True

    def _reset_state(self) -> None:
        self.context.reset()
        self._consecutive_duplicate_count = 0
        self._last_tool_call_signature = None
        self._stop_requested = False
        self._consecutive_timeouts = 0

    def run_messages(self) -> str:
        """运行 ReAct 循环（消息已由 build_initial_context 或调用方预设）。

        循环结束后由本方法负责将本轮 messages 提取为 session_history，
        调用方无需手工调用 extract_session_history。
        """
        # 工具定义在循环内不变，缓存引用以利用 LLM 客户端的 _normalize_tools 缓存
        tools = self.tool_manager.get_definitions()

        try:
            # ── ReAct 主循环：LLM 生成 → 工具执行 → 下一轮 ──
            while not self.context.iteration >= self.max_iterations:
                # 外部停止信号（如用户 Ctrl+C 或钩子请求）
                if self._stop_requested:
                    return self._stopped_message()

                self.context.advance_iteration()

                # 阶段 1：LLM 生成前——执行钩子（如审计注入），可中止循环
                hook_msg = self._run_hooks(HookPoint.PRE_GENERATE)
                self._emit_event(EventType.PRE_GENERATE, iteration=self.context.iteration,
                           messages=self.context.messages,
                           tools=tools,
                           hook_result=hook_msg)
                if hook_msg is not None:
                    return hook_msg

                # 阶段 2：调用 LLM 生成响应
                try:
                    response = self.llm.generate(
                        self.context.messages,
                        tools=tools,
                    )
                except LLMTimeoutError as e:
                    # 超时重试：注入提示让 LLM 简化或继续，最多重试 2 次
                    self._consecutive_timeouts += 1
                    logger.warning("LLM timeout (iteration %d): %s", self.context.iteration, e)
                    if self._consecutive_timeouts > self._max_timeout_retries:
                        msg = f"LLM 连续 {self._consecutive_timeouts} 次超时，已终止。"
                        self.context.append(Message(role="assistant", content=msg))
                        return msg
                    self.context.append(Message(
                        role="user",
                        content="[SYSTEM] 上次 LLM 请求超时。如有工具结果请继续执行，否则用更简洁的表述重试。",
                    ))
                    continue
                except LLMConnectionError as e:
                    # 连接失败（如网络不可用、API Key 无效）——不可恢复，终止循环
                    logger.error("LLM connection failed (iteration %d): %s", self.context.iteration, e)
                    result = f"LLM API 连接失败: {e}"
                    self.context.append(Message(role="assistant", content=result))
                    return result
                self._consecutive_timeouts = 0  # 成功一次即重置超时计数
                self.context.append(response)

                # 阶段 3：LLM 生成后——执行钩子，可中止循环
                hook_msg = self._run_hooks(HookPoint.POST_GENERATE, response=response)
                self._emit_event(EventType.POST_GENERATE, iteration=self.context.iteration, response=response,
                           hook_result=hook_msg)
                if hook_msg is not None:
                    return hook_msg or response.content or ""

                # 阶段 4：无工具调用 → 纯文本回复，循环结束
                if not response.tool_calls:
                    return response.content or ""

                # 阶段 5：重复工具调用检测——连续 3 次相同签名则终止
                if self._check_repetition(response):
                    return self._stuck_in_loop_message()

                # 阶段 6：逐个执行工具调用
                for tool_call in response.tool_calls:
                    result_msg = self._execute_one_tool(tool_call)
                    if result_msg is not None:
                        self.context.append(result_msg)

                # 工具结果已追加，压缩上下文供下一轮迭代使用
                self.context.usage = response.usage
                self.context.compress_if_needed()

            # 迭代预算耗尽
            hook_msg = self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
            self._emit_event(EventType.ON_BUDGET_EXHAUSTED, iteration=self.context.iteration,
                       max_iterations=self.max_iterations,
                       hook_result=hook_msg)
            return self._budget_exhausted_message()
        except KeyboardInterrupt:
            return self._stopped_message()
        finally:
            # 状态归属收敛：循环结束后统一提取 session_history，
            # 避免调用方（Agent）手工补位导致遗漏或不一致
            self.context.extract_session_history()

    def stream_messages(self) -> Iterator[StreamChunk]:
        """以流模式运行 ReAct 循环（消息已由 build_initial_context 或调用方预设）。

        生成器结束（耗尽或被关闭）时由本方法负责提取 session_history。
        """
        # 工具定义在循环内不变，缓存引用以利用 LLM 客户端的 _normalize_tools 缓存
        tools = self.tool_manager.get_definitions()

        try:
            while not self.context.iteration >= self.max_iterations:
                if self._stop_requested:
                    msg = self._stopped_message()
                    yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
                    return

                self.context.advance_iteration()

                hook_msg = self._run_hooks(HookPoint.PRE_GENERATE)
                self._emit_event(EventType.PRE_GENERATE, iteration=self.context.iteration,
                           messages=self.context.messages,
                           tools=tools,
                           hook_result=hook_msg)
                if hook_msg is not None:
                    yield StreamChunk(content=hook_msg, is_final=True)
                    return

                sess = StreamSession(self.llm.stream(
                    self.context.messages,
                    tools=tools,
                ))

                for chunk in sess.iter_chunks():
                    # 在 LLM 流式块之间检查停止信号。
                    # 即使 KeyboardInterrupt 被延迟传递（Windows  blocked I/O），
                    # 一旦 read timeout 突破阻塞后也能在此处迅速响应。
                    if self._stop_requested:
                        sess.request_stop()
                        msg = self._stopped_message()
                        yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
                        return
                    yield chunk

                response = sess.message()

                hook_msg = self._run_hooks(HookPoint.POST_GENERATE, response=response)
                self._emit_event(EventType.POST_GENERATE, iteration=self.context.iteration,
                           response=response, hook_result=hook_msg)
                if hook_msg is not None:
                    yield StreamChunk(content=hook_msg or response.content or "", is_final=True)
                    return

                self.context.append(response)

                if not sess.has_tool_calls:
                    # 纯文本回复：LLM 流的结束标记已被吸收，由循环层补发关闭当前轮
                    yield StreamChunk(is_final=True)
                    return

                if self._check_repetition(response):
                    yield StreamChunk(content=self._stuck_in_loop_message(), is_final=True)
                    return

                for tool_call in (response.tool_calls or []):
                    if self._stop_requested:
                        msg = self._stopped_message()
                        yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
                        return

                    result_msg = self._execute_one_tool(tool_call)
                    if result_msg is not None:
                        self.context.append(result_msg)
                        yield StreamChunk(
                            tool_result=result_msg,
                            is_final=False,
                        )

                # 本轮工具执行完毕：循环层主动发出结束标记关闭当前渲染轮，
                # 使下一轮 LLM 生成开启新的"第 N 次请求"标题。
                # LLM 流自带的结束标记已被上方吸收，轮次边界由此处统一管控。
                yield StreamChunk(is_final=True)

                # 工具结果已追加，压缩上下文供下一轮迭代使用
                self.context.usage = response.usage
                self.context.compress_if_needed()

            hook_msg = self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
            self._emit_event(EventType.ON_BUDGET_EXHAUSTED, iteration=self.context.iteration,
                       max_iterations=self.max_iterations,
                       hook_result=hook_msg)
            yield StreamChunk(content=self._budget_exhausted_message(), is_final=True)
        except KeyboardInterrupt:
            msg = self._stopped_message()
            yield StreamChunk(content="\n⏹️ " + msg, is_final=True)
            return
        finally:
            # 状态归属收敛：生成器结束时统一提取 session_history
            self.context.extract_session_history()

    # ── 工具执行 ──

    def _execute_one_tool(self, tool_call) -> Message | None:
        """执行单个工具调用：权限检查、钩子、事件、执行、审计。"""
        tool_name = tool_call.function.name

        # 阶段 1：权限检查——拒绝的回调反馈给 LLM，让其不再尝试
        if not self.permission_mgr.check(tool_call):
            return Message(
                role="tool",
                content="[Permission Denied] 用户拒绝了此工具调用，请勿再尝试使用此工具。",
                tool_call_id=tool_call.id,
            )

        # 阶段 2：工具执行前钩子——先执行钩子，再触发事件（事件携带钩子结果供观察者使用）
        hook_msg = self._run_hooks(HookPoint.PRE_TOOL, tool_call=tool_call)
        self._emit_event(EventType.PRE_TOOL, iteration=self.context.iteration, tool_call=tool_call,
                   hook_result=hook_msg)
        if hook_msg is not None:
            # 钩子跳过工具：不触发 POST_TOOL（工具未执行），
            # 跳过事实由 PRE_TOOL 的 hook_result 表达
            return Message(role="tool", content=hook_msg or "[Tool Skipped by Hook]", tool_call_id=tool_call.id)

        # 阶段 3：实际执行——记录耗时，用于后续指标
        duration = 0
        start = time.monotonic()
        try:
            result = self.tool_manager.execute(tool_call)
            duration = (time.monotonic() - start) * 1000
            content = result.content or ""

            # 阶段 4：工具执行后钩子——可请求终止循环
            hook_msg = self._run_hooks(HookPoint.POST_TOOL, tool_call=tool_call, tool_result=result)
            self._emit_event(EventType.POST_TOOL, iteration=self.context.iteration,
                       tool_name=tool_name, duration=duration, success=True, result=result,
                       hook_result=hook_msg)
            if hook_msg is not None:
                # 钩子返回非空消息说明它要求终止循环（如审计阻断、预算超限），
                # 设置停止标志使循环在下一个迭代边界结束。
                self._stop_requested = True

            return Message(
                role="tool",
                content=content,
                tool_call_id=getattr(result, "tool_call_id", tool_call.id),
            )

        except Exception as e:
            # 阶段 5：异常处理——记录错误事件，钩子可请求终止
            hook_msg = self._run_hooks(HookPoint.ON_ERROR, error=e)
            self._emit_event(EventType.POST_TOOL, iteration=self.context.iteration,
                       tool_name=tool_name, duration=duration, success=False, error=str(e))
            self._emit_event(EventType.ON_ERROR, iteration=self.context.iteration,
                       component="tool", error=str(e), hook_result=hook_msg)
            if hook_msg is not None:
                self._stop_requested = True

            # 将错误信息包装为 tool 消息返回给 LLM，让模型决定下一步
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
            self.context.append(Message(
                role="user",
                content="[SYSTEM] 检测到重复的工具调用。你正在重复调用相同的工具，请立即停止并回复用户。",
            ))

        if self._consecutive_duplicate_count >= 3:
            hook_msg = self._run_hooks(HookPoint.ON_BUDGET_EXHAUSTED)
            self._emit_event(EventType.ON_BUDGET_EXHAUSTED, iteration=self.context.iteration,
                       max_iterations=self.max_iterations, reason="duplicate_tool_calls",
                       hook_result=hook_msg)
            return True

        return False

    # ── 辅助方法 ──

    def _budget_exhausted_message(self) -> str:
        return f"Agent 在 {self.context.iteration} 次迭代后未完成。"

    def _stuck_in_loop_message(self) -> str:
        return (
            f"Agent 在第 {self.context.iteration} 次迭代检测到重复工具调用"
            f"（连续 {self._consecutive_duplicate_count} 次），已提前终止。"
        )

    def _stopped_message(self) -> str:
        self._stop_requested = False
        return f"Agent 在第 {self.context.iteration} 次迭代被终止。"

    def _run_hooks(self, point: HookPoint, **extra) -> str | None:
        """执行指定生命周期点的钩子。

        返回 abort 消息（钩子终止循环）或 None（继续执行）。

        已知字段传给 HookContext 命名参数，未知字段放入 extra 字典，
        避免因未知字段导致 TypeError（P0.2 回归）。
        """
        # 已知字段列表——HookContext 构造函数显式接受这些字段。
        # 不属于此列表的额外数据（如 iteration、max_iterations）自动归入 extra 字典，
        # 防止 HookContext 构造函数因未知 kwargs 抛 TypeError。
        known_fields = {"tool_call", "tool_result", "response", "error", "config"}
        context_kwargs = {}
        extra_data = {}
        for k, v in extra.items():
            if k in known_fields:
                context_kwargs[k] = v
            else:
                extra_data[k] = v
        context_kwargs["extra"] = extra_data

        context = HookContext(
            iteration=self.context.iteration,
            messages=self.context.messages,
            **context_kwargs,
        )
        for r in self.hook_manager.run(point, context):
            if r.action == HookAction.ABORT:
                return r.message
        return None

    def _emit_event(self, event_type: str, **data) -> None:
        self.event_emitter.fire(Event(
            type=event_type,
            iteration=self.context.iteration,
            data=data,
            timestamp=time.time(),
        ))