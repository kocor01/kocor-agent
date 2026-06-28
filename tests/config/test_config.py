"""测试配置加载"""

from __future__ import annotations

import os

from kocor.config import Config


class TestConfig:
    """测试 Config 默认值"""

    def setup_method(self):
        Config.reset()

    def test_default_provider(self):
        cfg = Config()
        assert cfg.provider == "openai"

    def test_default_max_iterations(self):
        cfg = Config()
        assert cfg.max_iterations == 20

    def test_default_timeout(self):
        cfg = Config()
        assert cfg.timeout == 300

    def test_custom_values(self):
        cfg = Config(
            provider="anthropic",
            max_iterations=10,
            timeout=60,
        )
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 10
        assert cfg.timeout == 60

    def test_load_returns_singleton(self):
        """Config.load() 返回全局单例，多次调用返回同一对象"""
        cfg1 = Config.load()
        cfg2 = Config.load()
        assert cfg1 is cfg2

    def test_load_after_reset_reloads(self):
        """Config.reset() 后 load() 重新加载"""
        cfg1 = Config.load()
        Config.reset()
        cfg2 = Config.load()
        assert cfg1 is not cfg2

    def test_reset_clears_instance(self):
        Config.load()
        Config.reset()
        assert Config._instance is None


class TestLoadConfig:
    """测试 _load 从环境变量读取"""

    def setup_method(self):
        for key in ["KOCOR_PROVIDER", "KOCOR_MAX_ITERATIONS", "KOCOR_TIMEOUT"]:
            os.environ.pop(key, None)

    def test_load_default(self):
        cfg = Config._load()
        assert cfg.provider == "openai"

    def test_load_from_env_provider(self):
        os.environ["KOCOR_PROVIDER"] = "anthropic"
        cfg = Config._load()
        assert cfg.provider == "anthropic"

    def test_load_from_env_provider_case_insensitive(self):
        """provider 大小写不敏感"""
        os.environ["KOCOR_PROVIDER"] = "OpenAI"
        cfg = Config._load()
        assert cfg.provider == "openai"

        os.environ["KOCOR_PROVIDER"] = "ANTHROPIC"
        cfg = Config._load()
        assert cfg.provider == "anthropic"

        os.environ["KOCOR_PROVIDER"] = "Anthropic"
        cfg = Config._load()
        assert cfg.provider == "anthropic"

    def test_load_from_env_max_iterations(self):
        os.environ["KOCOR_MAX_ITERATIONS"] = "15"
        cfg = Config._load()
        assert cfg.max_iterations == 15

    def test_load_from_env_timeout(self):
        os.environ["KOCOR_TIMEOUT"] = "45"
        cfg = Config._load()
        assert cfg.timeout == 45

    def test_load_all_from_env(self):
        os.environ["KOCOR_PROVIDER"] = "anthropic"
        os.environ["KOCOR_MAX_ITERATIONS"] = "30"
        os.environ["KOCOR_TIMEOUT"] = "60"
        cfg = Config._load()
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
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "不支持的 provider" in str(e) or "gemini" in str(e)

    def test_load_invalid_max_iterations_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_MAX_ITERATIONS"] = "abc"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_load_invalid_timeout_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_TIMEOUT"] = "-1"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass


class TestContextConfig:
    """测试上下文管理配置"""

    def setup_method(self):
        for key in [
            "KOCOR_CONTEXT_STRATEGY",
            "KOCOR_MEMORY_DIR",
            "KOCOR_CONTEXT_MAX_TOKENS",
            "KOCOR_PRESERVE_LAST_ROUNDS",
            "KOCOR_PRESERVE_FIRST_ROUNDS",
        ]:
            os.environ.pop(key, None)

    def test_default_context_strategy(self):
        cfg = Config()
        assert cfg.context_strategy == "default"

    def test_default_memory_dir(self):
        cfg = Config()
        assert cfg.memory_dir == ".kocor/memories"

    def test_default_context_max_tokens(self):
        cfg = Config()
        assert cfg.context_max_tokens == 200_000

    def test_default_preserve_last_rounds(self):
        cfg = Config()
        assert cfg.preserve_last_rounds == 3

    def test_load_context_strategy_from_env(self):
        os.environ["KOCOR_CONTEXT_STRATEGY"] = "sliding"
        cfg = Config._load()
        assert cfg.context_strategy == "sliding"

    def test_load_memory_dir_from_env(self):
        os.environ["KOCOR_MEMORY_DIR"] = ".kocor/memories"
        cfg = Config._load()
        assert cfg.memory_dir == ".kocor/memories"

    def test_load_context_max_tokens_from_env(self):
        os.environ["KOCOR_CONTEXT_MAX_TOKENS"] = "100000"
        cfg = Config._load()
        assert cfg.context_max_tokens == 100_000

    def test_load_preserve_last_rounds_from_env(self):
        os.environ["KOCOR_PRESERVE_LAST_ROUNDS"] = "5"
        cfg = Config._load()
        assert cfg.preserve_last_rounds == 5

    def test_default_preserve_first_rounds(self):
        cfg = Config()
        assert cfg.preserve_first_rounds == 1

    def test_load_preserve_first_rounds_from_env(self):
        os.environ["KOCOR_PRESERVE_FIRST_ROUNDS"] = "2"
        cfg = Config._load()
        assert cfg.preserve_first_rounds == 2


