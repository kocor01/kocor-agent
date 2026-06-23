"""HarnessEvent 和 EventEmitter 测试。"""

from kocor.harness.events import HarnessEvent, EventEmitter


class TestHarnessEvent:
    def test_create_event(self):
        event = HarnessEvent(
            type="pre_tool",
            iteration=1,
            data={"tool": "read_file"},
        )
        assert event.type == "pre_tool"
        assert event.iteration == 1
        assert event.data["tool"] == "read_file"
        assert event.timestamp > 0


class TestEventEmitter:
    def test_subscribe_and_fire(self):
        emitter = EventEmitter()
        received = []

        def handler(event):
            received.append(event)

        emitter.subscribe("pre_tool", handler)
        event = HarnessEvent(type="pre_tool", iteration=1, data={})
        emitter.fire(event)

        assert len(received) == 1
        assert received[0].type == "pre_tool"

    def test_multiple_subscribers(self):
        emitter = EventEmitter()
        results = []

        emitter.subscribe("pre_tool", lambda e: results.append("a"))
        emitter.subscribe("pre_tool", lambda e: results.append("b"))

        emitter.fire(HarnessEvent(type="pre_tool", iteration=1, data={}))
        assert results == ["a", "b"]

    def test_unsubscribe(self):
        emitter = EventEmitter()
        results = []

        def handler(event):
            results.append("called")

        emitter.subscribe("pre_tool", handler)
        emitter.unsubscribe("pre_tool", handler)
        emitter.fire(HarnessEvent(type="pre_tool", iteration=1, data={}))
        assert results == []

    def test_no_subscribers_does_not_error(self):
        emitter = EventEmitter()
        emitter.fire(HarnessEvent(type="unknown", iteration=1, data={}))

    def test_unregister_all(self):
        emitter = EventEmitter()
        emitter.subscribe("pre_tool", lambda e: None)
        emitter.subscribe("post_tool", lambda e: None)
        emitter.unregister_all()
        assert emitter._subscribers == {}

    def test_different_event_types_isolated(self):
        emitter = EventEmitter()
        pre = []
        post = []

        emitter.subscribe("pre_tool", lambda e: pre.append(e.type))
        emitter.subscribe("post_tool", lambda e: post.append(e.type))

        emitter.fire(HarnessEvent(type="pre_tool", iteration=1, data={}))
        assert pre == ["pre_tool"]
        assert post == []