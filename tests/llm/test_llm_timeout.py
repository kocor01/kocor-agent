"""LLM 超时/连接失败异常测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kocor.llm_provider.exceptions import LLMConnectionError, LLMTimeoutError
from kocor.llm_provider.llm_client import BaseLLMClient
from kocor.llm_provider.message import Message


class TimeoutTestClient(BaseLLMClient):
    """模拟超时的测试客户端。"""
    """模拟超时的测试客户端。"""

    def __init__(self, should_timeout=False, should_connect_fail=False):
        self.config = MagicMock()
        self.config.tool_timeout = 30
        self._tool_cache = None
        self._should_timeout = should_timeout
        self._should_connect_fail = should_connect_fail
        self._client = self._create_client()

    @property
    def provider(self) -> str:
        return "test"

    def _create_client(self):
        return object()

    def _normalize_in(self, messages):
        return []

    def _normalize_out(self, response, usage=None):
        return MagicMock()

    def _normalize_tool(self, tool):
        return {}

    def _api_generate(self, messages, tools, max_tokens, temperature, system=None):
        if self._should_timeout:
            from httpx import ReadTimeout
            raise ReadTimeout("timed out")
        if self._should_connect_fail:
            from httpx import ConnectError
            raise ConnectError("connection refused")
        return None

    def _api_stream(self, messages, tools, max_tokens, temperature, system=None):
        if self._should_timeout:
            from httpx import ReadTimeout
            raise ReadTimeout("timed out")
        if self._should_connect_fail:
            from httpx import ConnectError
            raise ConnectError("connection refused")
        return iter([])


class TestLLMTimeout:
    def test_timeout_raises_llm_timeout_error(self):
        client = TimeoutTestClient(should_timeout=True)
        with pytest.raises(LLMTimeoutError) as exc:
            client.generate([MagicMock()])
        assert "test" in str(exc.value)
        assert "timed out" in str(exc.value)

    def test_connect_failure_raises_llm_connection_error(self):
        client = TimeoutTestClient(should_connect_fail=True)
        with pytest.raises(LLMConnectionError) as exc:
            client.generate([MagicMock()])
        assert "test" in str(exc.value)

    def test_successful_call_does_not_raise(self):
        client = TimeoutTestClient(should_timeout=False)
        result = client.generate([MagicMock()])
        assert result is not None

    def test_timeout_error_distinct_from_keyboard_interrupt(self):
        """LLMTimeoutError 不是 KeyboardInterrupt 的子类。"""
        assert not issubclass(LLMTimeoutError, KeyboardInterrupt)
        assert not issubclass(LLMConnectionError, KeyboardInterrupt)