"""DebugManager 测试。"""

from kocor.harness.debug import DebugManager
from kocor.harness.events import HarnessEvent


class TestDebugManager:
    def test_disabled_by_default(self):
        dm = DebugManager()
        assert dm.enabled is False

    def test_records_events_when_enabled(self):
        dm = DebugManager(enabled=True)
        event = HarnessEvent(type="pre_tool", iteration=1, data={})
        dm.record_event(event)
        assert len(dm.events) == 1
        assert dm.events[0].type == "pre_tool"

    def test_does_not_record_when_disabled(self):
        dm = DebugManager(enabled=False)
        event = HarnessEvent(type="pre_tool", iteration=1, data={})
        dm.record_event(event)
        assert len(dm.events) == 0

    def test_clear_events(self):
        dm = DebugManager(enabled=True)
        dm.record_event(HarnessEvent(type="pre_tool", iteration=1, data={}))
        dm.clear()
        assert len(dm.events) == 0

    def test_multiple_events(self):
        dm = DebugManager(enabled=True)
        dm.record_event(HarnessEvent(type="pre_tool", iteration=1, data={}))
        dm.record_event(HarnessEvent(type="post_tool", iteration=1, data={}))
        assert len(dm.events) == 2

    def test_print_context_disabled(self, capsys):
        dm = DebugManager(enabled=False)
        dm.print_context([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_context_enabled(self, capsys):
        dm = DebugManager(enabled=True)
        dm.print_context([])
        captured = capsys.readouterr()
        assert "[DEBUG]" in captured.out