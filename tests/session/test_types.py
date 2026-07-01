"""测试会话数据模型。"""

from __future__ import annotations

from datetime import datetime

from kocor.session.types import SessionEntry, SessionResetPolicy


class TestSessionResetPolicy:
    """SessionResetPolicy 数据模型测试。"""

    def test_default_values(self):
        policy = SessionResetPolicy()
        assert policy.mode == "none"
        assert policy.idle_minutes == 1440
        assert policy.at_hour == 4

    def test_custom_values(self):
        policy = SessionResetPolicy(mode="idle", idle_minutes=60, at_hour=0)
        assert policy.mode == "idle"
        assert policy.idle_minutes == 60
        assert policy.at_hour == 0


class TestSessionEntry:
    """SessionEntry 数据模型测试。"""

    NOW = datetime(2026, 7, 2, 10, 0, 0)

    def test_create_entry(self):
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=self.NOW,
            updated_at=self.NOW,
        )
        assert entry.session_key == "kocor:default:cli"
        assert entry.session_id == "20260702_100000_a1b2c3d4"
        assert entry.message_count == 0
        assert entry.was_auto_reset is False

    def test_to_dict_roundtrip(self):
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=self.NOW,
            updated_at=self.NOW,
            message_count=5,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            was_auto_reset=True,
            auto_reset_reason="idle",
        )
        d = entry.to_dict()
        assert d["session_key"] == "kocor:default:cli"
        assert d["session_id"] == "20260702_100000_a1b2c3d4"
        assert d["message_count"] == 5
        assert d["was_auto_reset"] is True
        assert d["auto_reset_reason"] == "idle"

        restored = SessionEntry.from_dict(d)
        assert restored.session_key == entry.session_key
        assert restored.session_id == entry.session_id
        assert restored.created_at == entry.created_at
        assert restored.message_count == entry.message_count
        assert restored.was_auto_reset == entry.was_auto_reset
        assert restored.auto_reset_reason == entry.auto_reset_reason

    def test_from_dict_missing_fields(self):
        """缺失可选字段应有合理的默认值。"""
        now_str = self.NOW.isoformat()
        d = {
            "session_key": "kocor:default:cli",
            "session_id": "20260702_100000_a1b2c3d4",
            "created_at": now_str,
            "updated_at": now_str,
        }
        restored = SessionEntry.from_dict(d)
        assert restored.message_count == 0
        assert restored.was_auto_reset is False
        assert restored.auto_reset_reason is None
        assert restored.is_fresh_reset is False

    def test_is_fresh_reset_flag(self):
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=self.NOW,
            updated_at=self.NOW,
            is_fresh_reset=True,
        )
        d = entry.to_dict()
        restored = SessionEntry.from_dict(d)
        assert restored.is_fresh_reset is True

    def test_session_id_format(self):
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=self.NOW,
            updated_at=self.NOW,
        )
        # 格式：YYYYMMDD_HHMMSS_<8hex>
        parts = entry.session_id.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 8
        assert all(c in "0123456789abcdef" for c in parts[2])