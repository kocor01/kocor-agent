"""LLM Provider 异常定义。

与 KeyboardInterrupt 区分，允许上层（Loop）做重试或降级处理。
"""

from __future__ import annotations


class LLMTimeoutError(Exception):
    """LLM API 调用超时（非用户中断）。

    如 httpx.ReadTimeout、httpx.ConnectTimeout。
    上层可重试或降级，无需终止整个循环。
    """

    def __init__(
        self,
        provider: str = "unknown",
        timeout_seconds: int = 30,
        message: str | None = None,
    ):
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.message = message or f"{provider} API timed out after {timeout_seconds}s"
        super().__init__(self.message)


class LLMConnectionError(Exception):
    """LLM API 连接失败（非超时，如 DNS 解析失败、连接被拒绝）。

    通常意味着网络不可用或 API 端点配置错误，应终止而非重试。
    """

    def __init__(self, provider: str = "unknown", message: str | None = None):
        self.provider = provider
        self.message = message or f"{provider} API connection failed"
        super().__init__(self.message)