"""测试配置加载"""

import os

from kocor.config import LLMConfig, load_config


class TestLLMConfig:
    """测试 LLMConfig 默认值"""

    def test_default_provider(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"

    def test_default_max_iterations(self):
        cfg = LLMConfig()
        assert cfg.max_iterations == 20

    def test_default_timeout(self):
        cfg = LLMConfig()
        assert cfg.timeout == 30

    def test_custom_values(self):
        cfg = LLMConfig(
            provider="anthropic",
            max_iterations=10,
            timeout=60,
        )
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 10
        assert cfg.timeout == 60


class TestLoadConfig:
    """测试 load_config 从环境变量读取"""

    def setup_method(self):
        for key in ["KOCOR_PROVIDER", "KOCOR_MAX_ITERATIONS", "KOCOR_TIMEOUT"]:
            os.environ.pop(key, None)

    def test_load_default(self):
        cfg = load_config()
        assert cfg.provider == "openai"

    def test_load_from_env_provider(self):
        os.environ["KOCOR_PROVIDER"] = "anthropic"
        cfg = load_config()
        assert cfg.provider == "anthropic"

    def test_load_from_env_provider_case_insensitive(self):
        """provider 大小写不敏感"""
        os.environ["KOCOR_PROVIDER"] = "OpenAI"
        cfg = load_config()
        assert cfg.provider == "openai"

        os.environ["KOCOR_PROVIDER"] = "ANTHROPIC"
        cfg = load_config()
        assert cfg.provider == "anthropic"

        os.environ["KOCOR_PROVIDER"] = "Anthropic"
        cfg = load_config()
        assert cfg.provider == "anthropic"

    def test_load_from_env_max_iterations(self):
        os.environ["KOCOR_MAX_ITERATIONS"] = "15"
        cfg = load_config()
        assert cfg.max_iterations == 15

    def test_load_from_env_timeout(self):
        os.environ["KOCOR_TIMEOUT"] = "45"
        cfg = load_config()
        assert cfg.timeout == 45

    def test_load_all_from_env(self):
        os.environ["KOCOR_PROVIDER"] = "anthropic"
        os.environ["KOCOR_MAX_ITERATIONS"] = "30"
        os.environ["KOCOR_TIMEOUT"] = "60"
        cfg = load_config()
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 30
        assert cfg.timeout == 60


class TestLoadConfigValidation:
    """测试配置验证"""

    def setup_method(self):
        self._saved = {}
        for key in ["KOCOR_PROVIDER", "KOCOR_MAX_ITERATIONS", "KOCOR_TIMEOUT"]:
            self._saved[key] = os.environ.pop(key, None)

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def test_load_invalid_provider_raises(self):
        os.environ["KOCOR_PROVIDER"] = "gemini"
        try:
            load_config()
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "不支持的 provider" in str(e) or "gemini" in str(e)

    def test_load_invalid_max_iterations_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_MAX_ITERATIONS"] = "abc"
        try:
            load_config()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_load_invalid_timeout_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_TIMEOUT"] = "-1"
        try:
            load_config()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass
