"""测试 CLI 启动欢迎界面 _print_welcome。"""

from __future__ import annotations

import io
import re
from datetime import datetime
from unittest.mock import MagicMock, patch

from kocor.session.types import SessionEntry
from kocor.skill.types import InvokeStrategy


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列，方便断言纯文本内容。"""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def _make_skill(name: str, slash: bool = True) -> MagicMock:
    s = MagicMock()
    s.name = name
    s.invoke_strategy = InvokeStrategy.SLASH if slash else InvokeStrategy.NONE
    return s


class TestPrintWelcome:
    """测试 _print_welcome 启动显示。"""

    def _capture_welcome(self, session_manager=None, skill_manager=None) -> str:
        """捕获 _print_welcome 的全部输出（包括 Rich console.print）。"""
        from kocor.cli import _print_welcome

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_welcome(session_manager=session_manager, skill_manager=skill_manager)
        return buf.getvalue()

    def test_basic_header(self):
        """测试基本头部信息（品牌名、标语、版本）。"""
        output = self._capture_welcome()
        clean = _strip_ansi(output)

        assert "Kocor Agent" in clean
        assert "小而美的 LLM 自主 Agent 助手" in clean
        assert "V0.0.1" in clean
        assert "exit" in clean

    def test_with_session_continue(self):
        """测试有历史会话时显示继续会话信息。"""
        mock_session = MagicMock()
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_085549_1879d13f",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            title="你好",
            message_count=18,
        )
        mock_session.get_or_create_session.return_value = entry
        mock_session.store = MagicMock()

        output = self._capture_welcome(session_manager=mock_session)
        clean = _strip_ansi(output)

        assert "继续上次会话" in clean
        assert "你好" in clean
        assert "20260702_085549_1879d13f" in clean
        assert "18" in clean

    def test_with_session_new(self):
        """测试新会话显示。"""
        mock_session = MagicMock()
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_abcdef12",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            title="",
            message_count=0,
        )
        mock_session.get_or_create_session.return_value = entry
        mock_session.store = MagicMock()

        output = self._capture_welcome(session_manager=mock_session)
        clean = _strip_ansi(output)

        assert "新会话" in clean

    def test_with_session_auto_reset(self):
        """测试会话自动重置显示。"""
        mock_session = MagicMock()
        entry = SessionEntry(
            session_key="kocor:default:cli",
            session_id="20260702_100000_abcdef12",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            title="",
            message_count=0,
            was_auto_reset=True,
            auto_reset_reason="idle",
        )
        mock_session.get_or_create_session.return_value = entry
        mock_session.store = MagicMock()

        output = self._capture_welcome(session_manager=mock_session)
        clean = _strip_ansi(output)

        assert "会话已重置" in clean

    def test_with_skills(self):
        """测试 Slash 命令显示。"""
        mock_skill_mgr = MagicMock()
        mock_skill_mgr.list_skills.return_value = [
            _make_skill("joke"),
            _make_skill("weather"),
            _make_skill("uuid-gen"),
        ]

        output = self._capture_welcome(skill_manager=mock_skill_mgr)
        clean = _strip_ansi(output)

        assert "/joke" in clean
        assert "/weather" in clean
        assert "/uuid-gen" in clean

    def test_no_session(self):
        """测试无会话管理器时不显示会话信息。"""
        output = self._capture_welcome()
        clean = _strip_ansi(output)

        assert "会话" not in clean
