"""运行时指标收集器——通过事件订阅收集运行时指标。

不持有任何 Agent/LLM/ToolManager 引用，纯数据聚合。
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from kocor.event.event_manager import Event, EventType


class MetricsCollector:
    """运行时指标收集器。

    通过订阅事件系统收集性能指标和用量数据。
    可独立实例化测试，不需要 Agent 或其他组件。

    收集的指标：
    - 工具调用耗时（按工具名分组）
    - Token 用量（按迭代记录）
    - 错误计数（按组件分组）
    """

    def __init__(self):
        # 工具调用耗时（毫秒）
        self.tool_call_durations: dict[str, list[float]] = defaultdict(list)
        self.tool_call_count: dict[str, int] = defaultdict(int)
        self.tool_call_errors: dict[str, int] = defaultdict(int)

        # Token 用量
        self.token_usage: list[dict[str, Any]] = []

        # 错误
        self.error_count: int = 0
        self.errors_by_component: dict[str, int] = defaultdict(int)

        # 轮次指标
        self.iteration_count: int = 0
        self._current_iteration: int = 0
        self._iteration_start: float = 0.0
        self.iteration_durations: list[float] = []

    # ── 事件处理 ──

    def on_pre_generate(self, event: Event) -> None:
        """LLM 生成前：记录轮次开始时间。"""
        self._current_iteration = event.iteration
        self._iteration_start = time.monotonic()

    def on_post_generate(self, event: Event) -> None:
        """LLM 生成后：记录 Token 用量和轮次耗时。"""
        response = event.data.get("response")
        if response and response.usage:
            self.token_usage.append({
                "iteration": event.iteration,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_tokens": response.usage.cached_tokens,
            })

        # 记录轮次耗时
        if self._iteration_start > 0:
            elapsed = (time.monotonic() - self._iteration_start) * 1000  # ms
            self.iteration_durations.append(elapsed)

        self.iteration_count = event.iteration

    def on_post_tool(self, event: Event) -> None:
        """工具执行后：记录耗时和结果。"""
        data = event.data
        tool_name = data.get("tool_name", "unknown")

        self.tool_call_count[tool_name] += 1

        if data.get("success"):
            duration = data.get("duration")
            if duration and duration > 0:
                self.tool_call_durations[tool_name].append(duration)
        else:
            self.tool_call_errors[tool_name] += 1

    def on_error(self, event: Event) -> None:
        """错误事件：记录错误的组件。"""
        self.error_count += 1
        component = event.data.get("component", "unknown")
        self.errors_by_component[component] += 1

    # ── 报告 ──

    def report(self) -> dict[str, Any]:
        """生成当前指标摘要报告。

        Returns:
            包含关键统计信息的字典，可序列化为 JSON。
        """
        avg_durations = {}
        for name, vals in self.tool_call_durations.items():
            avg_durations[name] = sum(vals) / len(vals)

        return {
            "iterations": self.iteration_count,
            "avg_iteration_duration_ms": (
                sum(self.iteration_durations) / len(self.iteration_durations)
                if self.iteration_durations else 0
            ),
            "tool_calls": dict(self.tool_call_count),
            "avg_tool_duration_ms": avg_durations,
            "tool_errors": dict(self.tool_call_errors),
            "error_count": self.error_count,
            "errors_by_component": dict(self.errors_by_component),
            "total_api_calls": len(self.token_usage),
            "total_prompt_tokens": sum(t.get("prompt_tokens", 0) for t in self.token_usage),
            "total_completion_tokens": sum(t.get("completion_tokens", 0) for t in self.token_usage),
            "total_cached_tokens": sum(t.get("cached_tokens", 0) for t in self.token_usage),
        }

    def reset(self) -> None:
        """清除所有收集的指标。"""
        self.tool_call_durations.clear()
        self.tool_call_count.clear()
        self.tool_call_errors.clear()
        self.token_usage.clear()
        self.error_count = 0
        self.errors_by_component.clear()
        self.iteration_count = 0
        self.iteration_durations.clear()
        self._iteration_start = 0.0