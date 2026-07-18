"""日志订阅者 — 将事件写入日志文件。"""

import logging

from kocor.event.event_manager import Event, EventType
from kocor.logger import Logger


class Logs:
    """日志记录处理函数集合，供 EventSubscribe 注册。

    通过依赖注入接收 Logger 实例，不依赖全局单例。
    """

    def __init__(self, logger: Logger):
        self._logger = logger

    def _handle_pre_generate(self, event: Event) -> None:
        """记录 PRE_GENERATE 事件到日志。"""
        self._logger.event(EventType.PRE_GENERATE, logging.DEBUG, **event.data)

    def _handle_post_generate(self, event: Event) -> None:
        """记录 POST_GENERATE 事件到日志（含 token 消耗）。"""
        self._logger.event(EventType.POST_GENERATE, logging.DEBUG, **event.data)

    def _handle_pre_tool(self, event: Event) -> None:
        """记录 PRE_TOOL 事件到日志。"""
        self._logger.event(EventType.PRE_TOOL, logging.DEBUG, **event.data)

    def _handle_post_tool(self, event: Event) -> None:
        """记录 POST_TOOL 事件到日志。"""
        self._logger.event(EventType.POST_TOOL, logging.DEBUG, **event.data)

    def _handle_on_error(self, event: Event) -> None:
        """记录 ON_ERROR 事件到日志（ERROR 级别）。"""
        self._logger.event(EventType.ON_ERROR, logging.ERROR, **event.data)

    def _handle_on_budget_exhausted(self, event: Event) -> None:
        """记录 ON_BUDGET_EXHAUSTED 事件到日志（WARNING 级别）。"""
        self._logger.event(EventType.ON_BUDGET_EXHAUSTED, logging.WARNING, **event.data)