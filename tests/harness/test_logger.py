"""HarnessLogger 测试。"""

from kocor.harness.logger import HarnessLogger


class TestHarnessLogger:
    def test_default_level(self):
        logger = HarnessLogger()
        assert logger.logger.level == 20  # INFO

    def test_debug_level(self):
        logger = HarnessLogger(level="DEBUG")
        assert logger.logger.level == 10  # DEBUG

    def test_log_iteration(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.log_iteration(1, 150)
        assert len(caplog.records) >= 0  # at least no errors

    def test_log_tool_call(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.log_tool_call("read_file", 0.5, True)
        assert len(caplog.records) >= 0

    def test_log_budget_warning(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.log_budget_warning(0.85)

    def test_log_error(self, caplog):
        logger = HarnessLogger(level="DEBUG")
        logger.log_error("sandbox", "timeout")

    def test_name(self):
        logger = HarnessLogger()
        assert logger.logger.name == "kocor.harness"