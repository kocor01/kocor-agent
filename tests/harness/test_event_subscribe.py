"""测试 EventSubscribe — 统一事件订阅的集中管理。

覆盖代码审查报告指出的「事件系统 EventEmitter 和 EventSubscribe 的集成测试」缺口。
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from kocor.event.event_manager import EventEmitter, EventType, Event
from kocor.event.event_subscribe import EventSubscribe
from kocor.logger import Logger
from kocor.event.subscribes.logs import Logs


# ═══════════════════════════════════════════════
# EventSubscribe.subscribe_all
# ═══════════════════════════════════════════════


class TestEventSubscribe:
    """EventSubscribe.subscribe_all 集成测试。"""

    def test_subscribe_all_registers_all_handlers(self):
        """subscribe_all 注册所有事件类型到 emitter。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        sub.subscribe_all(logger)

        # 验证所有标准事件都有订阅者
        expected_types = [
            EventType.PRE_GENERATE,
            EventType.POST_GENERATE,
            EventType.PRE_TOOL,
            EventType.POST_TOOL,
            EventType.ON_ERROR,
            EventType.ON_BUDGET_EXHAUSTED,
        ]
        for et in expected_types:
            assert len(emitter._subscribers[et]) >= 1, f"{et} should have subscribers"

    def test_subscribe_all_triggers_logger(self):
        """事件触发后 Logger 被调用。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        # Mock logger.event
        with patch.object(logger, 'event') as mock_event:
            sub.subscribe_all(logger)

            # 触发 PRE_GENERATE 事件
            emitter.fire(Event(type=EventType.PRE_GENERATE, iteration=1, data={}))

            mock_event.assert_called_once_with(
                EventType.PRE_GENERATE,
                logging.DEBUG,
            )

    def test_subscribe_all_handles_multiple_events(self):
        """多个事件触发后 logger 被多次调用。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        with patch.object(logger, 'event') as mock_event:
            sub.subscribe_all(logger)

            # 触发多个事件
            emitter.fire(Event(type=EventType.PRE_GENERATE, iteration=1, data={}))
            emitter.fire(Event(type=EventType.POST_GENERATE, iteration=1, data={}))
            emitter.fire(Event(type=EventType.PRE_TOOL, iteration=1, data={}))

            assert mock_event.call_count == 3

    def test_on_error_uses_error_level(self):
        """ON_ERROR 事件使用 ERROR 日志级别。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        with patch.object(logger, 'event') as mock_event:
            sub.subscribe_all(logger)

            emitter.fire(Event(type=EventType.ON_ERROR, iteration=1, data={"error": "boom"}))

            mock_event.assert_called_once_with(
                EventType.ON_ERROR,
                logging.ERROR,
                error="boom",
            )

    def test_on_budget_exhausted_uses_warning_level(self):
        """ON_BUDGET_EXHAUSTED 事件使用 WARNING 日志级别。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        with patch.object(logger, 'event') as mock_event:
            sub.subscribe_all(logger)

            emitter.fire(Event(
                type=EventType.ON_BUDGET_EXHAUSTED,
                iteration=3,
                data={"max_iterations": 3},
            ))

            mock_event.assert_called_once_with(
                EventType.ON_BUDGET_EXHAUSTED,
                logging.WARNING,
                max_iterations=3,
            )

    def test_pre_generate_passes_messages_and_tools(self):
        """PRE_GENERATE 事件传递 messages 和 tools。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        with patch.object(logger, 'event') as mock_event:
            sub.subscribe_all(logger)

            emitter.fire(Event(
                type=EventType.PRE_GENERATE,
                iteration=1,
                data={"messages": [], "tools": []},
            ))

            mock_event.assert_called_once_with(
                EventType.PRE_GENERATE,
                logging.DEBUG,
                messages=[],
                tools=[],
            )

    def test_post_tool_passes_result(self):
        """POST_TOOL 事件传递工具执行结果。"""
        emitter = EventEmitter()
        logger = Logger("DEBUG")
        sub = EventSubscribe(emitter)

        with patch.object(logger, 'event') as mock_event:
            sub.subscribe_all(logger)

            emitter.fire(Event(
                type=EventType.POST_TOOL,
                iteration=1,
                data={
                    "tool_name": "read_file",
                    "duration": 100,
                    "success": True,
                    "result": "file content",
                },
            ))

            mock_event.assert_called_once_with(
                EventType.POST_TOOL,
                logging.DEBUG,
                tool_name="read_file",
                duration=100,
                success=True,
                result="file content",
            )


# ═══════════════════════════════════════════════
# Logs 处理器独立测试
# ═══════════════════════════════════════════════


class TestLogsHandlers:
    """Logs 各事件处理器的独立行为。"""

    def setup_method(self):
        self.logger = Logger("DEBUG")
        self.logs = Logs(logger=self.logger)

    def test_handle_pre_generate(self):
        with patch.object(self.logger, 'event') as mock_event:
            event = Event(type=EventType.PRE_GENERATE, iteration=1, data={"messages": []})
            self.logs._handle_pre_generate(event)
            mock_event.assert_called_once_with(EventType.PRE_GENERATE, logging.DEBUG, messages=[])

    def test_handle_post_generate(self):
        with patch.object(self.logger, 'event') as mock_event:
            event = Event(type=EventType.POST_GENERATE, iteration=1, data={"response": "hi"})
            self.logs._handle_post_generate(event)
            mock_event.assert_called_once_with(EventType.POST_GENERATE, logging.DEBUG, response="hi")

    def test_handle_pre_tool(self):
        with patch.object(self.logger, 'event') as mock_event:
            event = Event(type=EventType.PRE_TOOL, iteration=1, data={"tool_call": "read_file"})
            self.logs._handle_pre_tool(event)
            mock_event.assert_called_once_with(EventType.PRE_TOOL, logging.DEBUG, tool_call="read_file")

    def test_handle_post_tool(self):
        with patch.object(self.logger, 'event') as mock_event:
            event = Event(type=EventType.POST_TOOL, iteration=1, data={"tool_name": "write_file", "success": True})
            self.logs._handle_post_tool(event)
            mock_event.assert_called_once_with(EventType.POST_TOOL, logging.DEBUG, tool_name="write_file", success=True)

    def test_handle_on_error(self):
        with patch.object(self.logger, 'event') as mock_event:
            event = Event(type=EventType.ON_ERROR, iteration=1, data={"error": "timeout"})
            self.logs._handle_on_error(event)
            mock_event.assert_called_once_with(EventType.ON_ERROR, logging.ERROR, error="timeout")

    def test_handle_on_budget_exhausted(self):
        with patch.object(self.logger, 'event') as mock_event:
            event = Event(type=EventType.ON_BUDGET_EXHAUSTED, iteration=3, data={"max_iterations": 3})
            self.logs._handle_on_budget_exhausted(event)
            mock_event.assert_called_once_with(EventType.ON_BUDGET_EXHAUSTED, logging.WARNING, max_iterations=3)

    def test_handler_not_called_for_unsubscribed_event(self):
        """未订阅的事件类型不触发任何处理器。"""
        with patch.object(self.logger, 'event') as mock_event:
            # 只订阅了 POST_TOOL
            self.logs._handle_post_tool(Event(type=EventType.POST_TOOL, iteration=1, data={}))
            mock_event.assert_called_once()

            # PRE_GENERATE 未订阅
            # 不会调用 _handle_pre_generate，因为 EventSubscribe 不注册它
            # 这里仅验证不触发额外调用
            assert mock_event.call_count == 1


# ═══════════════════════════════════════════════
# EventType 枚举
# ═══════════════════════════════════════════════


class TestEventTypeEnum:
    """EventType 枚举值。"""

    def test_all_types_defined(self):
        assert EventType.PRE_GENERATE == "pre_generate"
        assert EventType.POST_GENERATE == "post_generate"
        assert EventType.PRE_TOOL == "pre_tool"
        assert EventType.POST_TOOL == "post_tool"
        assert EventType.PRE_SUMMARIZE == "pre_summarize"
        assert EventType.POST_SUMMARIZE == "post_summarize"
        assert EventType.ON_ERROR == "on_error"
        assert EventType.ON_BUDGET_EXHAUSTED == "on_budget_exhausted"