"""Kocor 运行时结构化日志。"""

import dataclasses
import json
import logging
import os
from datetime import date

from kocor.event.event_manager import EventType


class _EventEncoder(json.JSONEncoder):
    """将 dataclass 和 __dict__ 对象递归序列化为 JSON。"""

    def default(self, obj):
        """将 dataclass 和 __dict__ 对象递归序列化为 JSON。"""
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


class Logger:
    """为 kocor 操作提供结构化日志。

    内部两个 logger，各写各的文件：
      - ``info()/debug()/warning()/error()/event()`` → ``kocor.log``
      - ``audit()`` → ``audit.log``

    这是一个普通对象，非单例。通过依赖注入传递给消费者。
    """

    def __init__(self, level: str = "INFO", log_dir: str = "./log"):
        # 按日期分类日志目录，每天一个子目录
        today = date.today().isoformat()
        daily_dir = os.path.join(log_dir, today)
        os.makedirs(daily_dir, exist_ok=True)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # 默认日志 → kocor.log（运行时日志，含事件、错误等）
        self._default_logger = logging.getLogger("kocor")
        self._default_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self._default_logger.handlers:
            handler = logging.FileHandler(
                os.path.join(daily_dir, "kocor.log"), encoding="utf-8",
            )
            handler.setFormatter(formatter)
            self._default_logger.addHandler(handler)

        # 审计日志 → audit.log（独立 logger，禁止向 "kocor" 传播避免重复写入）
        self._audit_logger = logging.getLogger("kocor.audit")
        self._audit_logger.setLevel(logging.INFO)
        self._audit_logger.propagate = False
        if not self._audit_logger.handlers:
            handler = logging.FileHandler(
                os.path.join(daily_dir, "audit.log"), encoding="utf-8",
            )
            handler.setFormatter(formatter)
            self._audit_logger.addHandler(handler)

    # ── 公共方法 ──

    def event(self, event_type: EventType, level: int = logging.INFO, **data) -> None:
        """按事件类型记录日志到 kocor.log，消息体为 JSON 格式。"""
        self._default_logger.log(
            level, "%s %s",
            event_type.value,
            json.dumps(data, ensure_ascii=False, cls=_EventEncoder),
        )

    def debug(self, message: str) -> None:
        """写入 DEBUG 级别的运行时日志。"""
        self._default_logger.debug(message)

    def info(self, message: str) -> None:
        """写入 INFO 级别的运行时日志。"""
        self._default_logger.info(message)

    def warning(self, message: str) -> None:
        """写入 WARNING 级别的运行时日志。"""
        self._default_logger.warning(message)

    def error(self, message: str) -> None:
        """写入 ERROR 级别的运行时日志。"""
        self._default_logger.error(message)

    def audit(self, message: str) -> None:
        """记录审计日志到 audit.log。"""
        self._audit_logger.info(message)