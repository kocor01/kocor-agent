"""统一管理事件订阅。"""

from __future__ import annotations

from kocor.event.event_manager import EventEmitter, EventType
from kocor.event.subscribes.logs import Logs
from kocor.event.subscribes.metrics import MetricsCollector
from kocor.logger import Logger


class EventSubscribe:
    """集中管理标准事件订阅，避免在入口点分散订阅逻辑。"""

    def __init__(self, event_emitter: EventEmitter):
        self._emitter = event_emitter

    def subscribe_all(
        self,
        logger: Logger,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """订阅所有标准事件处理器。

        包括日志记录器和可选的指标收集器。

        Args:
            logger: Logger 实例，注入到日志订阅处理器中。
            metrics: 可选的 MetricsCollector 实例，注入到指标订阅处理器中。
        """
        handler = Logs(logger=logger)

        self._emitter.subscribe(EventType.PRE_GENERATE, handler._handle_pre_generate)
        self._emitter.subscribe(EventType.POST_GENERATE, handler._handle_post_generate)
        self._emitter.subscribe(EventType.PRE_TOOL, handler._handle_pre_tool)
        self._emitter.subscribe(EventType.POST_TOOL, handler._handle_post_tool)
        self._emitter.subscribe(EventType.ON_ERROR, handler._handle_on_error)
        self._emitter.subscribe(EventType.ON_BUDGET_EXHAUSTED, handler._handle_on_budget_exhausted)

        if metrics is not None:
            self._emitter.subscribe(EventType.PRE_GENERATE, metrics.on_pre_generate)
            self._emitter.subscribe(EventType.POST_GENERATE, metrics.on_post_generate)
            self._emitter.subscribe(EventType.POST_TOOL, metrics.on_post_tool)
            self._emitter.subscribe(EventType.ON_ERROR, metrics.on_error)