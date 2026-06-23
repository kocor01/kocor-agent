"""Harness 运行时的事件系统。"""

import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class HarnessEvent:
    """通过观察者和钩子分发的运行时事件。"""

    type: str
    iteration: int
    data: dict
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class EventEmitter:
    """简单的发布-订阅事件总线。

    按事件类型持有订阅者回调。HookRunner 在此基础上构建了钩子生命周期，
    但任何组件都可以直接订阅。
    """

    def __init__(self):
        self._subscribers: dict[str, list[callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: callable) -> None:
        """为某个事件类型注册一个处理函数。"""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """移除某个事件类型的处理函数。"""
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    def fire(self, event: HarnessEvent) -> None:
        """将事件分发给所有已注册的处理函数。"""
        for handler in self._subscribers.get(event.type, []):
            handler(event)

    def unregister_all(self) -> None:
        """移除所有订阅者。"""
        self._subscribers.clear()