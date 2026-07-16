"""SecretStr 安全字符串容器测试 + API Key 自动包装测试。"""

from __future__ import annotations

import os

from kocor._secret import SecretStr
from kocor.config import Config


class TestSecretStr:
    def test_secret_str_hides_value(self):
        s = SecretStr("sk-abc123xyz")
        assert "sk-abc123xyz" not in repr(s)
        assert "sk-abc123xyz" not in str(s)
        assert "******" in repr(s)

    def test_secret_str_reveal(self):
        s = SecretStr("sk-abc123xyz")
        assert s.reveal() == "sk-abc123xyz"

    def test_secret_str_equality(self):
        assert SecretStr("sk-abc") == SecretStr("sk-abc")
        assert SecretStr("sk-abc") == "sk-abc"
        assert SecretStr("sk-abc") != SecretStr("sk-xyz")
        assert SecretStr("sk-abc") != "sk-xyz"

    def test_secret_str_bool(self):
        assert not SecretStr("")
        assert SecretStr("sk-abc")


class TestApiKeyAutoWrap:
    def test_api_key_auto_wrapped(self):
        """ConfigLoader 自动包装 API Key 字段为 SecretStr。"""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        Config.reset()
        cfg = Config.load()
        assert isinstance(cfg.openai_api_key, SecretStr)
        assert cfg.openai_api_key.reveal() == "sk-test-key"
        del os.environ["OPENAI_API_KEY"]

    def test_repr_does_not_leak_key(self):
        """Config 的 repr() 不应泄露 API Key。"""
        cfg = Config(
            openai_api_key=SecretStr("sk-leaked"),
            anthropic_api_key=SecretStr("sk-ant-leaked"),
        )
        r = repr(cfg)
        assert "sk-leaked" not in r
        assert "sk-ant-leaked" not in r