"""测试会话键生成。"""

from __future__ import annotations

from kocor.config import Config
from kocor.session.manager import SessionManager
from kocor.session.store import SessionStore


class TestSessionKey:
    """SessionManager.session_key 测试。"""

    def test_default_key(self):
        """无 profile 时应使用 Config.session_name 的默认值"default"。"""
        sm = SessionManager(store=SessionStore())
        assert sm.session_key == "kocor:default:cli"

    def test_named_profile(self):
        sm = SessionManager(store=SessionStore(), profile="project-x")
        assert sm.session_key == "kocor:project-x:cli"

    def test_empty_profile_falls_to_default(self):
        sm = SessionManager(store=SessionStore(), profile="")
        assert sm.session_key == "kocor:default:cli"

    def test_config_session_name(self):
        """profile 为 None 时应从 Config.session_name 读取。"""
        Config.load().session_name = "env-test"
        try:
            sm = SessionManager(store=SessionStore())
            assert sm.session_key == "kocor:env-test:cli"
        finally:
            Config.reset()

    def test_explicit_profile_overrides_config(self):
        """显式传入 profile 应优先于 Config.session_name。"""
        Config.load().session_name = "from-config"
        try:
            sm = SessionManager(store=SessionStore(), profile="explicit")
            assert sm.session_key == "kocor:explicit:cli"
        finally:
            Config.reset()