"""子代理运行器：构建、执行、汇总。"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import TYPE_CHECKING, Any

from kocor.config import Config
from kocor.event.event_manager import Event, EventType
from kocor.tools.toolsets.subagent.child_builder import assemble_child_loop
from kocor.tools.toolsets.subagent.summary import extract_summary

if TYPE_CHECKING:
    from kocor.event.event_manager import EventEmitter
    from kocor.llm_provider.llm_client import LLMClient
    from kocor.loop import Loop
    from kocor.tools.tool_manager import ToolManager


class SubagentRunner:
    """子代理运行器。

    构造时注入父依赖，run() 时构建隔离的子 Loop 并执行。
    depth 为当前代理在委托树中的深度（0=顶层）。
    嵌套的子代理会构造新的 SubagentRunner（depth+1）并注册到其 ToolManager。
    """

    def __init__(
        self,
        parent_llm: LLMClient,
        parent_tool_manager: ToolManager,
        parent_event_emitter: EventEmitter,
        depth: int = 0,
        max_depth: int | None = None,
    ):
        self._parent_llm = parent_llm
        self._parent_tm = parent_tool_manager
        self._parent_emitter = parent_event_emitter
        self._depth = depth
        self._max_depth = max_depth if max_depth is not None else Config.load().subagent_max_depth
        # 批量中断事件：设此标志时批处理中断等待中的子代理
        self._stop_requested = threading.Event()
        self._stop_requested.clear()
        # 跟踪运行中的子 Loop，供 stop() 传播中断信号
        self._running_loops: list[Loop] = []

    def stop(self) -> None:
        """请求中断所有运行中的子代理。"""
        self._stop_requested.set()
        # 停止所有正在运行的子 Loop（每个 Loop 检查 _stop_requested）
        for loop in self._running_loops:
            loop.stop()

    def run(
        self,
        goal: str | None = None,
        context: str | None = None,
        tasks: list[dict] | None = None,
    ) -> dict[str, Any]:
        """执行子代理任务。

        Args:
            goal: 单任务目标（与 tasks 二选一）
            context: 传给子代理的背景信息
            tasks: 批量并行子任务（与 goal 二选一）

        Returns:
            结构化结果字典
        """
        if goal and tasks:
            return {"status": "error", "summary": "不能同时提供 goal 和 tasks"}
        if not goal and not tasks:
            return {"status": "error", "summary": "必须提供 goal 或 tasks"}

        if tasks:
            return self._execute_batch(tasks)
        return self._run_single_child(goal or "", context)

    def _inject_orchestrator_runner(self, child_loop: "Loop") -> None:
        """为 orchestrator 子代理注入子级 SubagentRunner。

        替换 child_loop 的 ToolManager 中 subagent 工具的占位 handler，
        使其捕获子级 SubagentRunner，从而允许子代理递归委派孙代理。
        """
        child_tm = child_loop.tool_manager
        if "subagent" not in child_tm._handlers:
            return  # 安全防护：无 subagent 工具（不应发生）
        child_runner = SubagentRunner(
            parent_llm=self._parent_llm,
            parent_tool_manager=child_tm,
            parent_event_emitter=self._parent_emitter,
            depth=self._depth + 1,
            max_depth=self._max_depth,
        )
        from kocor.tools.toolsets.subagent.tool import SubagentTool
        child_tm._handlers["subagent"] = lambda **kw: SubagentTool.handler(runner=child_runner, **kw)

    def _run_single_child(self, goal: str, context: str | None) -> dict[str, Any]:
        """执行单个子代理并返回结构化结果。"""
        start = time.monotonic()
        child_loop = assemble_child_loop(
            goal=goal,
            context=context,
            parent_llm=self._parent_llm,
            parent_tool_manager=self._parent_tm,
            depth=self._depth,
            max_depth=self._max_depth,
        )

        # 若是 orchestrator 角色，注入子级 SubagentRunner 替换占位 handler
        if (self._depth + 1) < self._max_depth:
            self._inject_orchestrator_runner(child_loop)

        # 注册到运行中列表，供 stop() 传播中断信号
        self._running_loops.append(child_loop)

        # 发射开始事件
        self._parent_emitter.fire(Event(
            type=EventType.SUBAGENT_START,
            iteration=0,
            data={"subagent_id": f"sa-{id(child_loop):x}", "goal": goal, "depth": self._depth},
        ))

        try:
            subagent_timeout = Config.load().subagent_timeout
            if subagent_timeout > 0:
                # 有 wall-clock 超时：在守护线程中运行，超时则标记 timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(child_loop.run_messages)
                    final_text = fut.result(timeout=subagent_timeout)
            else:
                final_text = child_loop.run_messages()
            status = "completed"
            if child_loop.ctx.iteration >= child_loop.max_iterations:
                status = "budget_exhausted"
        except concurrent.futures.TimeoutError:
            final_text = None
            status = "timeout"
            context_data = {"subagent_id": f"sa-{id(child_loop):x}", "status": status, "error": f"timeout after {subagent_timeout}s"}
        except Exception as e:
            final_text = None
            status = "error"
            context_data = {"subagent_id": f"sa-{id(child_loop):x}", "status": status, "error": str(e)}
        else:
            context_data = {
                "subagent_id": f"sa-{id(child_loop):x}",
                "status": status,
                "duration": time.monotonic() - start,
                "usage": {
                    "prompt_tokens": getattr(child_loop.ctx.usage, "prompt_tokens", 0) if child_loop.ctx.usage else 0,
                    "completion_tokens": getattr(child_loop.ctx.usage, "completion_tokens", 0) if child_loop.ctx.usage else 0,
                },
            }

        # 发射完成事件
        self._parent_emitter.fire(Event(
            type=EventType.SUBAGENT_COMPLETE,
            iteration=0,
            data=context_data,
        ))

        result = extract_summary(final_text, status=status)
        result["duration"] = context_data.get("duration", time.monotonic() - start)
        result["iterations"] = child_loop.ctx.iteration
        return result

    def _execute_batch(self, tasks: list[dict]) -> dict[str, Any]:
        """批量并行执行多个子代理。"""
        max_concurrent = Config.load().subagent_max_concurrent
        if len(tasks) > max_concurrent:
            return {
                "status": "error",
                "summary": f"tasks 数量 {len(tasks)} 超过上限 {max_concurrent}，请减少批量或拆分多次调用",
            }

        results: list[dict | None] = [None] * len(tasks)
        completed_count = 0

        def run_one(index: int, t: dict) -> dict:
            return self._run_single_child(
                goal=t.get("goal", ""),
                context=t.get("context"),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            future_map = {
                pool.submit(run_one, i, t): i
                for i, t in enumerate(tasks)
            }

            pending = set(future_map.keys())
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for f in done:
                    idx = future_map[f]
                    try:
                        results[idx] = f.result()
                        completed_count += 1
                        goal = tasks[idx].get("goal", "")
                        dur = results[idx].get("duration", 0)
                        print(f"[{completed_count}/{len(tasks)}] {goal[:40]} ({dur:.1f}s)")
                    except Exception as e:
                        results[idx] = {"status": "error", "summary": str(e)}

                if self._stop_requested.is_set():
                    # 中断：取消剩余 pending
                    for f in pending:
                        f.cancel()
                        idx = future_map[f]
                        results[idx] = {"status": "interrupted", "summary": ""}
                    break

        # 去掉 None（不应发生）
        completed = [r for r in results if r is not None]
        return {
            "results": completed,
            "total_duration": sum(r.get("duration", 0) for r in completed),
        }