"""测试会话重置策略。"""

from __future__ import annotations

from datetime import datetime, timedelta

from kocor.session.reset_policy import should_reset
from kocor.session.types import SessionEntry, SessionResetPolicy


class TestShouldReset:
    """should_reset() 重置策略评估测试。"""

    NOW = datetime(2026, 7, 2, 14, 30, 0)  # 下午 2:30

    def make_entry(self, updated_at: datetime | None = None) -> SessionEntry:
        return SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_a1b2c3d4",
            created_at=self.NOW,
            updated_at=updated_at or self.NOW,
        )

    def test_mode_none_never_resets(self):
        """mode=none 应永不重置。"""
        old = self.NOW - timedelta(days=30)
        entry = self.make_entry(updated_at=old)
        policy = SessionResetPolicy(mode="none")
        assert should_reset(entry, policy, now=self.NOW) is None

    def test_idle_not_expired(self):
        """空闲未超时不应重置。"""
        entry = self.make_entry(updated_at=self.NOW - timedelta(minutes=30))
        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        assert should_reset(entry, policy, now=self.NOW) is None

    def test_idle_expired(self):
        """超过空闲时间应重置。"""
        entry = self.make_entry(updated_at=self.NOW - timedelta(minutes=90))
        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        assert should_reset(entry, policy, now=self.NOW) == "idle"

    def test_idle_exactly_at_deadline(self):
        """恰好等于空闲时长边界不应重置（deadline 是 strict >）。"""
        entry = self.make_entry(updated_at=self.NOW - timedelta(minutes=60))
        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        # 每分钟检查一次，此时刚刚好 60 分钟过去，now > deadline 应为 False
        assert should_reset(entry, policy, now=self.NOW) is None

    def test_daily_before_reset_hour(self):
        """当日凌晨活动在上次重置之后，不应重置。"""
        now = datetime(2026, 7, 2, 3, 0, 0)  # 凌晨 3 点，重置在 4 点
        entry = self.make_entry(updated_at=datetime(2026, 7, 2, 1, 0, 0))  # 凌晨 1 点有活动（在昨日的重置点之后）
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        # activity(01:00) > today_reset(昨日04:00) → 在重置之后有活动
        assert should_reset(entry, policy, now=now) is None

    def test_daily_crossed_reset_hour(self):
        """上次活动在当日重置时刻之前，且当前已过重置时刻，应重置。"""
        now = datetime(2026, 7, 2, 5, 0, 0)  # 凌晨 5 点，已过 4 点重置
        entry = self.make_entry(updated_at=datetime(2026, 7, 2, 3, 0, 0))  # 凌晨 3 点有活动（在 4 点前）
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        # activity(03:00) < today_reset(今日04:00) → 活动在今日重置前
        assert should_reset(entry, policy, now=now) == "daily"

    def test_daily_after_reset_hour_updated_today(self):
        """在每日重置时刻之后且已有当日活动，不应重置。"""
        now = datetime(2026, 7, 2, 10, 0, 0)  # 上午 10 点
        entry = self.make_entry(updated_at=datetime(2026, 7, 2, 8, 0, 0))  # 早上 8 点有活动
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        assert should_reset(entry, policy, now=now) is None

    def test_daily_after_reset_hour_no_activity(self):
        """跨过每日重置时刻且无当日活动应重置。"""
        now = datetime(2026, 7, 2, 10, 0, 0)  # 上午 10 点
        entry = self.make_entry(updated_at=datetime(2026, 7, 1, 22, 0, 0))  # 昨晚 10 点
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        assert should_reset(entry, policy, now=now) == "daily"

    def test_custom_idle_minutes(self):
        """自定义空闲分钟数应生效。"""
        entry = self.make_entry(updated_at=self.NOW - timedelta(minutes=5))
        policy = SessionResetPolicy(mode="idle", idle_minutes=10)
        assert should_reset(entry, policy, now=self.NOW) is None

        entry2 = self.make_entry(updated_at=self.NOW - timedelta(minutes=15))
        assert should_reset(entry2, policy, now=self.NOW) == "idle"

    def test_custom_at_hour(self):
        """自定义每日重置时刻应生效。"""
        now = datetime(2026, 7, 2, 9, 0, 0)
        entry = self.make_entry(updated_at=datetime(2026, 7, 1, 23, 0, 0))
        policy = SessionResetPolicy(mode="daily", at_hour=8)  # 早上 8 点重置
        assert should_reset(entry, policy, now=now) == "daily"

    def test_daily_not_expired_midnight_cross(self):
        """跨天但未到重置时刻不应重置。"""
        # 昨天 23:00 有活动，现在是凌晨 2:00，重置在 4:00
        now = datetime(2026, 7, 3, 2, 0, 0)
        entry = self.make_entry(updated_at=datetime(2026, 7, 2, 23, 0, 0))
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        # updated_at(23:00) > today_reset(04:00 昨天) → 在重置之后有活动，不触发
        assert should_reset(entry, policy, now=now) is None