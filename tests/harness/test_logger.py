"""HarnessLogger 测试。"""

from kocor.harness.event.event_manager import EventType
from kocor.harness.logger import HarnessLogger


class TestHarnessLogger:
    def test_default_level(self):
        logger = HarnessLogger()
        assert logger.logger.level == 20  # INFO

    def test_debug_level(self):
        logger = HarnessLogger(level="DEBUG")
        assert logger.logger.level == 10  # DEBUG

    def test_event_info_level(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.event(EventType.POST_GENERATE, iteration=1, token_count=150)
        assert len(caplog.records) >= 1
        assert "post_generate" in caplog.text
        assert "【iteration】=1" in caplog.text

    def test_event_error_level(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.event(EventType.ON_ERROR, component="tool", error="timeout")
        assert len(caplog.records) >= 1

    def test_event_warning_level(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.event(EventType.ON_BUDGET_EXHAUSTED, iteration=5, budget_ratio=1.0)
        assert len(caplog.records) >= 1

    def test_info_shortcut(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.info("hello world")
        assert "hello world" in caplog.text

    def test_warning_shortcut(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.warning("disk full")
        assert "disk full" in caplog.text

    def test_error_shortcut(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.error("something broke")
        assert "something broke" in caplog.text

    def test_name(self):
        logger = HarnessLogger()
        assert logger.logger.name == "kocor.harness"
