"""Harness 运行时的结构化日志。"""

import logging


class HarnessLogger:
    """为 harness 操作提供结构化日志。

    包装标准库的 logging 模块，提供针对迭代、工具调用、预算和错误的便捷方法。
    """

    def __init__(self, level: str = "INFO", log_path: str | None = None):
        self.logger = logging.getLogger("kocor.harness")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8") if log_path else logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            self.logger.addHandler(handler)

    def log_iteration(self, iteration: int, token_count: int) -> None:
        self.logger.info("iteration=%d tokens=%d", iteration, token_count)

    def log_tool_call(
        self, name: str, duration_ms: float, success: bool
    ) -> None:
        self.logger.info("tool=%s duration_ms=%.0f success=%s", name, duration_ms, success)

    def log_budget_warning(self, ratio: float) -> None:
        self.logger.warning("budget_usage=%.0f%%", ratio * 100)

    def log_error(self, component: str, error: str) -> None:
        self.logger.error("component=%s error=%s", component, error)
