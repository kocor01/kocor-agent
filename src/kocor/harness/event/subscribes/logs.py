"""日志订阅者 — 将事件写入日志文件。"""

from kocor.harness.event.event_manager import EventType, HarnessEvent
from kocor.harness.logger import get_logger


class Logs:
    """日志记录处理函数集合，供 EventSubscribe 注册。"""

    def _handle_pre_generate(self, event: HarnessEvent) -> None:
        get_logger().event(EventType.PRE_GENERATE, **event.data)

    def _handle_post_generate(self, event: HarnessEvent) -> None:
        get_logger().event(EventType.POST_GENERATE, **event.data)

    def _handle_pre_tool(self, event: HarnessEvent) -> None:
        get_logger().event(EventType.PRE_TOOL, **event.data)

    def _handle_post_tool(self, event: HarnessEvent) -> None:
        get_logger().event(EventType.POST_TOOL, **event.data)

    def _handle_on_error(self, event: HarnessEvent) -> None:
        get_logger().event(EventType.ON_ERROR, **event.data)

    def _handle_on_budget_exhausted(self, event: HarnessEvent) -> None:
        get_logger().event(EventType.ON_BUDGET_EXHAUSTED, **event.data)
