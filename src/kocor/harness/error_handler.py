"""错误处理和优雅降级策略。"""


RETRYABLE_ERRORS = {
    "RateLimitError", "Timeout", "ServiceUnavailableError", "InternalServerError",
}


class ErrorHandler:
    """Harness 错误处理策略。

    将错误分类为可重试（临时）或永久性错误，
    并支持指数退避重试。
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def handle_tool_error(
        self, error: Exception, tool_name: str, iteration: int
    ) -> str:
        """生成工具执行错误的用户可读消息。"""
        error_type = type(error).__name__

        if error_type in RETRYABLE_ERRORS:
            return (
                f"[重试] 工具 {tool_name} 遇到临时错误 ({error_type})，"
                f"请稍后重试"
            )
        if isinstance(error, PermissionError):
            return str(error)
        return f"Error executing {tool_name}: {error_type}: {error}"

    def handle_llm_error(
        self, error: Exception, retry_count: int
    ) -> tuple[bool, str]:
        """决定 LLM 错误后是否重试。

        返回 (should_retry, message)。
        """
        error_type = type(error).__name__

        if error_type in RETRYABLE_ERRORS and retry_count < self.max_retries:
            wait = 2 ** retry_count
            return (True, f"LLM 临时错误，{wait}s 后重试...")

        return (False, f"LLM 错误: {error}")


class GracefulDegradation:
    """达到限制时的优雅降级策略。"""

    def partial_result(self, tool_history: list) -> str:
        """在预算耗尽时生成摘要消息。"""
        if not tool_history:
            return "Agent 在完成任何操作前已达到限制。"

        lines = ["Agent 已达到执行限制。已完成的操作为："]
        for rec in tool_history:
            lines.append(f"  {rec.iteration}. {rec.tool_name}()")
        return "\n".join(lines)