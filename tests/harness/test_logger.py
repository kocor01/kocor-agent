"""Logger 测试。"""

import pytest

from kocor.harness.event.event_manager import EventType
from kocor.harness.logger import Logger, get_logger, setup_logger


class TestLogger:
    def test_default_level(self):
        logger = Logger()
        assert logger.logger.level == 20  # INFO

    def test_debug_level(self):
        logger = Logger(level="DEBUG")
        assert logger.logger.level == 10  # DEBUG

    def test_event_info_level(self, caplog):
        logger = Logger(level="DEBUG")
        logger.event(EventType.POST_GENERATE, iteration=1, token_count=150)
        assert len(caplog.records) >= 1
        assert "post_generate" in caplog.text
        assert "【iteration】=1" in caplog.text

    def test_event_error_level(self, caplog):
        logger = Logger(level="DEBUG")
        logger.event(EventType.ON_ERROR, component="tool", error="timeout")
        assert len(caplog.records) >= 1

    def test_event_warning_level(self, caplog):
        logger = Logger(level="DEBUG")
        logger.event(EventType.ON_BUDGET_EXHAUSTED, iteration=5, budget_ratio=1.0)
        assert len(caplog.records) >= 1

    def test_info_shortcut(self, caplog):
        logger = Logger(level="DEBUG")
        logger.info("hello world")
        assert "hello world" in caplog.text

    def test_warning_shortcut(self, caplog):
        logger = Logger(level="DEBUG")
        logger.warning("disk full")
        assert "disk full" in caplog.text

    def test_error_shortcut(self, caplog):
        logger = Logger(level="DEBUG")
        logger.error("something broke")
        assert "something broke" in caplog.text

    def test_name(self):
        logger = Logger()
        assert logger.logger.name == "kocor.harness"

    def test_setup_and_get_logger(self):
        logger = setup_logger("DEBUG", log_dir="./log")
        assert logger is get_logger()
        assert get_logger().logger.level == 10  # DEBUG

    def test_get_logger_before_setup_raises(self):
        from kocor.harness.logger import _logger_instance, get_logger
        saved = _logger_instance
        try:
            import kocor.harness.logger as mod
            mod._logger_instance = None
            with pytest.raises(RuntimeError, match="not initialized"):
                get_logger()
        finally:
            mod._logger_instance = saved
