"""Harness 运行时的结构化日志。"""

import logging
import os
from datetime import date

from kocor.harness.event.event_manager import EventType


class Logger:
    """为 harness 操作提供结构化日志。

    包装标准库的 logging 模块，提供事件驱动的日志记录。
    日志按天写入对应日期的子目录：实际写入路径为 ``{log_dir}/{日期}/kocor.log``。
    """

    _EVENT_LEVELS = {
        EventType.PRE_GENERATE: logging.INFO,
        EventType.POST_GENERATE: logging.INFO,
        EventType.PRE_TOOL: logging.INFO,
        EventType.POST_TOOL: logging.INFO,
        EventType.ON_ERROR: logging.ERROR,
        EventType.ON_BUDGET_EXHAUSTED: logging.WARNING,
    }

    def __init__(self, level: str = "INFO", log_dir: str = "./log"):
        today = date.today().isoformat()
        daily_dir = os.path.join(log_dir, today)
        os.makedirs(daily_dir, exist_ok=True)
        daily_path = os.path.join(daily_dir, "kocor.log")

        self.logger = logging.getLogger("kocor.harness")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.FileHandler(daily_path, encoding="utf-8")
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


_logger_instance: Logger | None = None


def setup_logger(level: str = "INFO", log_dir: str = "./log") -> Logger:
    """初始化并设置全局 Logger 实例。"""
    global _logger_instance
    _logger_instance = Logger(level, log_dir)
    return _logger_instance


def get_logger() -> Logger:
    """获取全局 Logger 实例，未初始化时抛出 RuntimeError。"""
    if _logger_instance is None:
        raise RuntimeError(
            "Logger not initialized, call setup_logger() first"
        )
    return _logger_instance
