"""统一管理事件订阅。"""

from kocor.harness.event.event_manager import EventEmitter, EventType
from kocor.harness.event.subscribes.logs import Logs


class EventSubscribe:
    """集中管理标准事件订阅，避免在入口点分散订阅逻辑。"""

    def __init__(self, event_emitter: EventEmitter):
        self._emitter = event_emitter

    def subscribe_all(self) -> None:
        """订阅所有标准日志记录事件。"""
        handler = Logs()

        self._emitter.subscribe(EventType.PRE_GENERATE, handler._handle_pre_generate)
        self._emitter.subscribe(EventType.POST_GENERATE, handler._handle_post_generate)
        self._emitter.subscribe(EventType.PRE_TOOL, handler._handle_pre_tool)
        self._emitter.subscribe(EventType.POST_TOOL, handler._handle_post_tool)
        self._emitter.subscribe(EventType.ON_ERROR, handler._handle_on_error)
        self._emitter.subscribe(EventType.ON_BUDGET_EXHAUSTED, handler._handle_on_budget_exhausted)
