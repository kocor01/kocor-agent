"""测试 cron 调度解析。"""

from __future__ import annotations

import pytest

from kocor.tools.toolsets.cron.jobs import HAS_CRONITER, parse_schedule


class TestParseSchedule:
    """测试 parse_schedule 调度解析。"""

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

    def test_invalid_duration_rejected(self):
        """时长格式如 '30m'、'2h' 不再支持。"""
        with pytest.raises(ValueError):
            parse_schedule("30m")
        with pytest.raises(ValueError):
            parse_schedule("2h")

    def test_invalid_interval_rejected(self):
        """间隔格式如 'every 30m' 不再支持。"""
        with pytest.raises(ValueError):
            parse_schedule("every 30m")

    def test_invalid_natural_daily_rejected(self):
        """自然语言格式如 'every day at 22:02' 不再支持。"""
        with pytest.raises(ValueError):
            parse_schedule("every day at 22:02")
        with pytest.raises(ValueError):
            parse_schedule("每天 22:02")

    def test_invalid_cron_expression(self):
        """无效的 cron 表达式字段。"""
        with pytest.raises(ValueError):
            parse_schedule("a b c d e")