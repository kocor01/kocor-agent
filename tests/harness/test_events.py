"""HarnessEvent 和 EventEmitter 测试。"""

from kocor.harness.events import HarnessEvent, EventEmitter, EventType


class TestHarnessEvent:
    def test_create_event(self):
        event = HarnessEvent(
            type=EventType.PRE_TOOL,
            iteration=1,
            data={"tool": "read_file"},
        )
        assert event.type == EventType.PRE_TOOL
        assert event.iteration == 1
        assert event.data["tool"] == "read_file"
        assert event.timestamp > 0


class TestEventEmitter:
    def test_subscribe_and_fire(self):
        emitter = EventEmitter()
        received = []

        def handler(event):
            received.append(event)

        emitter.subscribe(EventType.PRE_TOOL, handler)
        event = HarnessEvent(type=EventType.PRE_TOOL, iteration=1, data={})
        emitter.fire(event)

        assert len(received) == 1
        assert received[0].type == EventType.PRE_TOOL

    def test_multiple_subscribers(self):
        emitter = EventEmitter()
        results = []

        emitter.subscribe(EventType.PRE_TOOL, lambda e: results.append("a"))
        emitter.subscribe(EventType.PRE_TOOL, lambda e: results.append("b"))

        emitter.fire(HarnessEvent(type=EventType.PRE_TOOL, iteration=1, data={}))
        assert results == ["a", "b"]

    def test_unsubscribe(self):
        emitter = EventEmitter()
        results = []

        def handler(event):
            results.append("called")

        emitter.subscribe(EventType.PRE_TOOL, handler)
        emitter.unsubscribe(EventType.PRE_TOOL, handler)
        emitter.fire(HarnessEvent(type=EventType.PRE_TOOL, iteration=1, data={}))
        assert results == []

    def test_no_subscribers_does_not_error(self):
        emitter = EventEmitter()
        emitter.fire(HarnessEvent(type="unknown", iteration=1, data={}))

    def test_unregister_all(self):
        emitter = EventEmitter()
        emitter.subscribe(EventType.PRE_TOOL, lambda e: None)
        emitter.subscribe(EventType.POST_TOOL, lambda e: None)
        emitter.unregister_all()
        assert emitter._subscribers == {}

    def test_different_event_types_isolated(self):
        emitter = EventEmitter()
        pre = []
        post = []

        emitter.subscribe(EventType.PRE_TOOL, lambda e: pre.append(e.type))
        emitter.subscribe(EventType.POST_TOOL, lambda e: post.append(e.type))

        emitter.fire(HarnessEvent(type=EventType.PRE_TOOL, iteration=1, data={}))
        assert pre == [EventType.PRE_TOOL]
        assert post == []