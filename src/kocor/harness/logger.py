"""Harness 运行时的结构化日志。"""

import logging

from kocor.harness.event.event_manager import EventType


class HarnessLogger:
    """为 harness 操作提供结构化日志。

    包装标准库的 logging 模块，提供事件驱动的日志记录。
    """

    _EVENT_LEVELS = {
        EventType.PRE_GENERATE: logging.INFO,
        EventType.POST_GENERATE: logging.INFO,
        EventType.PRE_TOOL: logging.INFO,
        EventType.POST_TOOL: logging.INFO,
        EventType.ON_ERROR: logging.ERROR,
        EventType.ON_BUDGET_EXHAUSTED: logging.WARNING,
    }

    def __init__(self, level: str = "INFO", log_path: str = "./log/kocor.log"):
        self.logger = logging.getLogger("kocor.harness")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8") if log_path else logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            self.logger.addHandler(handler)

    def event(self, event_type: EventType, **data) -> None:
        """按事件类型记录日志，自动选择日志级别。"""
        level = self._EVENT_LEVELS.get(event_type, logging.INFO)
        parts = " ".join(f"【{k}】={v}" for k, v in data.items())
        self.logger.log(level, "%s %s", event_type.value, parts)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)
