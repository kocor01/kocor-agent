"""ErrorHandler 和 GracefulDegradation 测试。"""

from kocor.harness.error_handler import ErrorHandler


class RateLimitError(Exception): ...
class Timeout(Exception): ...
class ServiceUnavailableError(Exception): ...


class TestErrorHandler:
    def test_retryable_error_says_retry(self):
        handler = ErrorHandler(max_retries=3)
        should_retry, msg = handler.handle_llm_error(
            RateLimitError("rate limited"), retry_count=0
        )
        assert should_retry is True
        assert "重试" in msg

    def test_non_retryable_error(self):
        handler = ErrorHandler(max_retries=3)
        should_retry, msg = handler.handle_llm_error(
            Exception("AuthError"), retry_count=0
        )
        assert should_retry is False

    def test_timeout_error_is_retryable(self):
        handler = ErrorHandler(max_retries=3)
        should_retry, msg = handler.handle_llm_error(
            Timeout("timeout"), retry_count=0
        )
        assert should_retry is True

    def test_rate_limit_error_is_retryable(self):
        handler = ErrorHandler(max_retries=3)
        should_retry, msg = handler.handle_llm_error(
            RateLimitError("rate limit"), retry_count=0
        )
        assert should_retry is True

    def test_exceeds_max_retries(self):
        handler = ErrorHandler(max_retries=3)
        should_retry, msg = handler.handle_llm_error(
            RateLimitError("rate limit"), retry_count=3
        )
        assert should_retry is False

    def test_handle_tool_error_retryable(self):
        handler = ErrorHandler()
        msg = handler.handle_tool_error(
            RateLimitError("rate limit"), "read_file", 1
        )
        assert "重试" in msg

    def test_handle_tool_error_normal(self):
        handler = ErrorHandler()
        msg = handler.handle_tool_error(
            Exception("ValueError: bad arg"), "read_file", 1
        )
        assert "Error executing" in msg


