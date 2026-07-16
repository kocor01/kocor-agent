"""MetricsCollector 指标收集器测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from kocor.event.event_manager import Event, EventType
from kocor.event.subscribes.metrics import MetricsCollector
from kocor.llm_provider.message import Usage


class TestMetricsCollector:
    def test_tracks_tool_duration(self):
        collector = MetricsCollector()
        collector.on_post_tool(Event(
            type="post_tool", iteration=1,
            data={"tool_name": "bash", "duration": 1500.0, "success": True},
        ))
        assert collector.tool_call_durations["bash"] == [1500.0]

    def test_tracks_multiple_tool_calls(self):
        collector = MetricsCollector()
        for i in range(3):
            collector.on_post_tool(Event(
                type="post_tool", iteration=1,
                data={"tool_name": "read", "duration": 100.0 * (i + 1), "success": True},
            ))
        assert collector.tool_call_count["read"] == 3
        assert sum(collector.tool_call_durations["read"]) == 600.0

    def test_tracks_token_usage(self):
        collector = MetricsCollector()
        collector.on_post_generate(Event(
            type="post_generate", iteration=1,
            data={
                "response": MagicMock(
                    usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=20)
                ),
            },
        ))
        assert collector.token_usage[0]["prompt_tokens"] == 100
        assert collector.token_usage[0]["completion_tokens"] == 50

    def test_tracks_errors(self):
        collector = MetricsCollector()
        collector.on_error(Event(
            type="on_error", iteration=1,
            data={"component": "tool", "error": "Timeout"},
        ))
        collector.on_error(Event(
            type="on_error", iteration=1,
            data={"component": "tool", "error": "Timeout"},
        ))
        collector.on_error(Event(
            type="on_error", iteration=1,
            data={"component": "llm", "error": "RateLimit"},
        ))
        assert collector.error_count == 3
        assert collector.errors_by_component["tool"] == 2
        assert collector.errors_by_component["llm"] == 1

    def test_report_returns_summary(self):
        collector = MetricsCollector()
        collector.on_post_tool(Event(
            type="post_tool", iteration=1,
            data={"tool_name": "bash", "duration": 1000.0, "success": True},
        ))
        collector.on_post_generate(Event(
            type="post_generate", iteration=1,
            data={
                "response": MagicMock(
                    usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=20)
                ),
            },
        ))
        report = collector.report()
        assert report["tool_calls"]["bash"] == 1
        assert report["total_prompt_tokens"] == 100

    def test_reset_clears_all(self):
        collector = MetricsCollector()
        collector.on_post_tool(Event(
            type="post_tool", iteration=1,
            data={"tool_name": "bash", "duration": 100.0, "success": True},
        ))
        collector.on_error(Event(
            type="on_error", iteration=1,
            data={"component": "tool", "error": "Timeout"},
        ))
        collector.reset()
        assert collector.error_count == 0
        assert len(collector.tool_call_durations) == 0
        assert collector.iteration_count == 0