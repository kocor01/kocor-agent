"""测试会话键生成。"""

from __future__ import annotations

import os

from kocor.session.session_key import build_session_key


class TestBuildSessionKey:
    """build_session_key() 测试。"""

    def test_default_key(self):
        """无 profile 时应使用"default"。"""
        key = build_session_key()
        assert key == "kocor:default:cli"

    def test_named_profile(self):
        key = build_session_key(profile="project-x")
        assert key == "kocor:project-x:cli"

    def test_empty_profile_falls_to_default(self):
        key = build_session_key(profile="")
        assert key == "kocor:default:cli"

    def test_default_profile_name(self):
        key = build_session_key(profile="default")
        assert key == "kocor:default:cli"

    def test_env_var_profile(self):
        """profile 为 None 时应从 KOCOR_SESSION_NAME 环境变量读取。"""
        os.environ["KOCOR_SESSION_NAME"] = "env-test"
        try:
            key = build_session_key()
            assert key == "kocor:env-test:cli"
        finally:
            del os.environ["KOCOR_SESSION_NAME"]

    def test_explicit_profile_overrides_env(self):
        """显式传入 profile 应优先于环境变量。"""
        os.environ["KOCOR_SESSION_NAME"] = "from-env"
        try:
            key = build_session_key(profile="explicit")
            assert key == "kocor:explicit:cli"
        finally:
            del os.environ["KOCOR_SESSION_NAME"]