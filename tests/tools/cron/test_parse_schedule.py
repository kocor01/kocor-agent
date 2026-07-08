"""测试 cron 调度解析。"""

from __future__ import annotations

import pytest

from kocor.tools.toolset.cron.jobs import HAS_CRONITER, parse_duration, parse_schedule


class TestParseDuration:
    """测试 parse_duration 时长解析。"""

    def test_minutes(self):
        assert parse_duration("30m") == 30
        assert parse_duration("30min") == 30
        assert parse_duration("30mins") == 30
        assert parse_duration("30minute") == 30
        assert parse_duration("30minutes") == 30

    def test_hours(self):
        assert parse_duration("2h") == 120
        assert parse_duration("2hr") == 120
        assert parse_duration("2hrs") == 120
        assert parse_duration("2hour") == 120
        assert parse_duration("2hours") == 120

    def test_days(self):
        assert parse_duration("1d") == 1440
        assert parse_duration("1day") == 1440
        assert parse_duration("1days") == 1440

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("")
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("10x")


class TestParseSchedule:
    """测试 parse_schedule 调度解析。"""

    def test_interval_every(self):
        """'every 30m' → 循环间隔 30 分钟。"""
        result = parse_schedule("every 30m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 30
        assert "30m" in result["display"]

    def test_interval_every_2h(self):
        """'every 2h' → 循环间隔 2 小时。"""
        result = parse_schedule("every 2h")
        assert result["kind"] == "interval"
        assert result["minutes"] == 120

    @pytest.mark.skipif(not HAS_CRONITER, reason="croniter not installed")
    def test_cron_expression_5_fields(self):
        """'0 9 * * *' → cron 表达式。"""
        result = parse_schedule("0 9 * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * *"

    @pytest.mark.skipif(not HAS_CRONITER, reason="croniter not installed")
    def test_cron_expression_complex(self):
        """'*/15 * * * *' → 每 15 分钟的 cron 表达式。"""
        result = parse_schedule("*/15 * * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "*/15 * * * *"

    def test_one_shot_duration(self):
        """纯时长如 '30m' → 一次性，从当前时间起 N 分钟后。"""
        result = parse_schedule("30m")
        assert result["kind"] == "once"
        assert result["run_at"] is not None
        assert "30m" in result["display"]

    def test_one_shot_duration_2h(self):
        result = parse_schedule("2h")
        assert result["kind"] == "once"
        assert result["run_at"] is not None

    def test_iso_timestamp(self):
        """'2026-07-08T14:00:00' → 一次性定时。"""
        result = parse_schedule("2026-07-08T14:00:00")
        assert result["kind"] == "once"
        assert "2026-07-08" in result["run_at"]

    def test_iso_timestamp_with_z(self):
        """带 Z 后缀的 ISO 时间戳。"""
        result = parse_schedule("2026-07-08T14:00:00Z")
        assert result["kind"] == "once"

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            parse_schedule("")

    def test_invalid_garbage(self):
        with pytest.raises(ValueError):
            parse_schedule("not_a_schedule")

    def test_invalid_cron_expression(self):
        """无效的 cron 表达式字段。"""
        with pytest.raises(ValueError):
            parse_schedule("a b c d e")