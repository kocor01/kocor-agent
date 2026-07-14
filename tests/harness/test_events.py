"""Event 和 EventEmitter 测试。"""

from kocor.event.event_manager import Event, EventEmitter, EventType


class TestEvent:
    def test_create_event(self):
        event = Event(
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
        event = Event(type=EventType.PRE_TOOL, iteration=1, data={})
        emitter.fire(event)

        assert len(received) == 1
        assert received[0].type == EventType.PRE_TOOL

    def test_multiple_subscribers(self):
        emitter = EventEmitter()
        results = []

        emitter.subscribe(EventType.PRE_TOOL, lambda e: results.append("a"))
        emitter.subscribe(EventType.PRE_TOOL, lambda e: results.append("b"))

        emitter.fire(Event(type=EventType.PRE_TOOL, iteration=1, data={}))
        assert results == ["a", "b"]

    def test_unsubscribe(self):
        emitter = EventEmitter()
        results = []

        def handler(event):
            results.append("called")

        emitter.subscribe(EventType.PRE_TOOL, handler)
        emitter.unsubscribe(EventType.PRE_TOOL, handler)
        emitter.fire(Event(type=EventType.PRE_TOOL, iteration=1, data={}))
        assert results == []

    def test_no_subscribers_does_not_error(self):
        emitter = EventEmitter()
        emitter.fire(Event(type="unknown", iteration=1, data={}))

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

        emitter.fire(Event(type=EventType.PRE_TOOL, iteration=1, data={}))
        assert pre == [EventType.PRE_TOOL]
        assert post == []


class TestSubagentEvents:
    """测试子代理事件类型。"""

    def test_subagent_start_event_type_exists(self):
        assert EventType.SUBAGENT_START == "subagent_start"

    def test_subagent_complete_event_type_exists(self):
        assert EventType.SUBAGENT_COMPLETE == "subagent_complete"

    def test_subagent_start_fire_and_receive(self):
        emitter = EventEmitter()
        received = []

        def handler(event):
            received.append(event)

        emitter.subscribe(EventType.SUBAGENT_START, handler)
        event = Event(
            type=EventType.SUBAGENT_START,
            iteration=0,
            data={"subagent_id": "sa-1", "goal": "test", "depth": 1},
        )
        emitter.fire(event)

        assert len(received) == 1
        assert received[0].data["subagent_id"] == "sa-1"
        assert received[0].data["depth"] == 1

    def test_subagent_complete_fire_with_usage(self):
        emitter = EventEmitter()
        received = []

        def handler(event):
            received.append(event)

        emitter.subscribe(EventType.SUBAGENT_COMPLETE, handler)
        event = Event(
            type=EventType.SUBAGENT_COMPLETE,
            iteration=0,
            data={
                "subagent_id": "sa-1",
                "status": "completed",
                "duration": 12.3,
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
            },
        )
        emitter.fire(event)

        assert len(received) == 1
        assert received[0].data["status"] == "completed"
        assert received[0].data["usage"]["prompt_tokens"] == 500