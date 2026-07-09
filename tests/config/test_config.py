"""测试配置加载"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from kocor.config import Config, _resolve_data_path


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

    def test_default_tool_timeout(self):
        cfg = Config()
        assert cfg.tool_timeout == 30

    def test_custom_values(self):
        cfg = Config(
            provider="anthropic",
            max_iterations=10,
        )
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 10

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
        for key in ["KOCOR_PROVIDER", "KOCOR_MAX_ITERATIONS", "KOCOR_TOOL_TIMEOUT"]:
            os.environ.pop(key, None)
        # 隔离本地 .env 文件：_load() 内部会调用 load_dotenv() 重新载入 .env，
        # 会把开发环境配置（如 KOCOR_PROVIDER）回填进 os.environ，污染默认值测试
        self._dotenv_patch = patch("kocor.config.load_dotenv")
        self._dotenv_patch.start()
        Config.reset()

    def teardown_method(self):
        self._dotenv_patch.stop()
        # 清除单例，避免本类的隔离配置泄漏到后续依赖 .env 的测试
        Config.reset()

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

    def test_load_tool_timeout_from_env(self):
        os.environ["KOCOR_TOOL_TIMEOUT"] = "60"
        cfg = Config._load()
        assert cfg.tool_timeout == 60

    def test_load_all_from_env(self):
        os.environ["KOCOR_PROVIDER"] = "anthropic"
        os.environ["KOCOR_MAX_ITERATIONS"] = "30"
        os.environ["KOCOR_TOOL_TIMEOUT"] = "45"
        cfg = Config._load()
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 30
        assert cfg.tool_timeout == 45


class TestLoadConfigValidation:
    """测试配置验证"""

    def setup_method(self):
        self._saved = {}
        for key in ["KOCOR_PROVIDER", "KOCOR_MAX_ITERATIONS", "KOCOR_TOOL_TIMEOUT"]:
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

    def test_load_invalid_tool_timeout_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_TOOL_TIMEOUT"] = "-1"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

class TestPermissionPolicyConfig:
    """测试权限策略配置"""

    def setup_method(self):
        Config.reset()
        os.environ.pop("KOCOR_PERMISSION_POLICY", None)

    def test_default_permission_policy(self):
        cfg = Config()
        assert cfg.permission_policy == "default"

    def test_custom_permission_policy(self):
        cfg = Config(permission_policy="strict")
        assert cfg.permission_policy == "strict"

    def test_load_permission_policy_from_env(self):
        os.environ["KOCOR_PERMISSION_POLICY"] = "permissive"
        cfg = Config._load()
        assert cfg.permission_policy == "permissive"

    def test_load_permission_policy_case_insensitive(self):
        os.environ["KOCOR_PERMISSION_POLICY"] = "Strict"
        cfg = Config._load()
        assert cfg.permission_policy == "strict"

    def test_load_invalid_permission_policy_raises(self):
        os.environ["KOCOR_PERMISSION_POLICY"] = "invalid"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "不支持的 permission_policy" in str(e)

    def test_cli_applies_to_config(self):
        """模拟 CLI 参数通过 Config.load() 修改配置"""
        Config.load().permission_policy = "strict"
        assert Config.load().permission_policy == "strict"


class TestMaxTokensConfig:
    """测试 max_tokens 配置（响应最大长度）"""

    def setup_method(self):
        Config.reset()
        os.environ.pop("KOCOR_MAX_TOKENS", None)

    def teardown_method(self):
        os.environ.pop("KOCOR_MAX_TOKENS", None)

    def test_default_max_tokens(self):
        cfg = Config()
        assert cfg.max_tokens == 50000

    def test_custom_max_tokens(self):
        cfg = Config(max_tokens=8192)
        assert cfg.max_tokens == 8192

    def test_load_max_tokens_from_env(self):
        os.environ["KOCOR_MAX_TOKENS"] = "8192"
        cfg = Config._load()
        assert cfg.max_tokens == 8192

    def test_load_max_tokens_invalid_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_MAX_TOKENS"] = "abc"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_load_max_tokens_negative_raises(self):
        os.environ["KOCOR_PROVIDER"] = "openai"
        os.environ["KOCOR_MAX_TOKENS"] = "-1"
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
            "KOCOR_LOG_DIR",
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
        assert cfg.memory_dir == _resolve_data_path(".kocor/memories")

    def test_load_memory_dir_from_env_absolute(self):
        os.environ["KOCOR_MEMORY_DIR"] = "/tmp/kocor_memories"
        cfg = Config._load()
        assert cfg.memory_dir == "/tmp/kocor_memories"

    def test_default_log_dir(self):
        cfg = Config()
        assert cfg.log_dir == "./log"

    def test_load_log_dir_from_env(self):
        os.environ["KOCOR_LOG_DIR"] = "./log"
        cfg = Config._load()
        assert cfg.log_dir == _resolve_data_path("./log")

    def test_load_log_dir_from_env_absolute(self):
        os.environ["KOCOR_LOG_DIR"] = "/tmp/kocor_logs"
        cfg = Config._load()
        assert cfg.log_dir == "/tmp/kocor_logs"

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


class TestFieldValidation:
    """测试配置项校验（阈值/限制/数值字段的边界与类型）。

    这些校验分支原本在 Config._load 中已存在，但缺少测试覆盖。
    作为重构前的行为锁定测试：先在当前实现上通过，重构后仍须保持。
    """

    # 重构中可能调整错误信息的 env（如 file_read_max_* 的裸 int() 报错），
    # 这类只断言抛 ValueError；其余断言保留错误信息关键字以锁行为。
    _VALIDATION_ENVS = [
        "KOCOR_PROVIDER", "KOCOR_CONTEXT_SUMMARY_THRESHOLD",
        "KOCOR_CONTEXT_TRUNCATE_THRESHOLD", "KOCOR_MEMORY_CHAR_LIMIT",
        "KOCOR_USER_CHAR_LIMIT", "KOCOR_NUDGE_INTERVAL",
        "KOCOR_FILE_READ_MAX_CHARS", "KOCOR_FILE_READ_MAX_LINES",
        "KOCOR_FILE_SEARCH_MAX_RESULTS", "KOCOR_FILE_SEARCH_TIMEOUT",
    ]

    def setup_method(self):
        self._dotenv_patch = patch("kocor.config.load_dotenv")
        self._dotenv_patch.start()
        self._saved = {k: os.environ.pop(k, None) for k in self._VALIDATION_ENVS}
        Config.reset()

    def teardown_method(self):
        self._dotenv_patch.stop()
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        Config.reset()

    # --- 上下文阈值 [0, 1] ---
    def test_summary_threshold_above_range_raises(self):
        os.environ["KOCOR_CONTEXT_SUMMARY_THRESHOLD"] = "1.5"
        with pytest.raises(ValueError):
            Config._load()

    def test_summary_threshold_below_range_raises(self):
        os.environ["KOCOR_CONTEXT_SUMMARY_THRESHOLD"] = "-0.1"
        with pytest.raises(ValueError):
            Config._load()

    def test_summary_threshold_non_numeric_raises(self):
        os.environ["KOCOR_CONTEXT_SUMMARY_THRESHOLD"] = "abc"
        with pytest.raises(ValueError):
            Config._load()

    def test_truncate_threshold_above_range_raises(self):
        os.environ["KOCOR_CONTEXT_TRUNCATE_THRESHOLD"] = "2.0"
        with pytest.raises(ValueError):
            Config._load()

    def test_truncate_threshold_below_range_raises(self):
        os.environ["KOCOR_CONTEXT_TRUNCATE_THRESHOLD"] = "-1"
        with pytest.raises(ValueError):
            Config._load()

    # --- 字符上限 >= 1 ---
    def test_memory_char_limit_zero_raises(self):
        os.environ["KOCOR_MEMORY_CHAR_LIMIT"] = "0"
        with pytest.raises(ValueError):
            Config._load()

    def test_memory_char_limit_non_int_raises(self):
        os.environ["KOCOR_MEMORY_CHAR_LIMIT"] = "abc"
        with pytest.raises(ValueError):
            Config._load()

    def test_user_char_limit_zero_raises(self):
        os.environ["KOCOR_USER_CHAR_LIMIT"] = "0"
        with pytest.raises(ValueError):
            Config._load()

    def test_user_char_limit_non_int_raises(self):
        os.environ["KOCOR_USER_CHAR_LIMIT"] = "x"
        with pytest.raises(ValueError):
            Config._load()

    # --- nudge_interval >= 0 ---
    def test_nudge_interval_negative_raises(self):
        os.environ["KOCOR_NUDGE_INTERVAL"] = "-1"
        with pytest.raises(ValueError):
            Config._load()

    def test_nudge_interval_non_int_raises(self):
        os.environ["KOCOR_NUDGE_INTERVAL"] = "abc"
        with pytest.raises(ValueError):
            Config._load()

    # --- 文件工具数值字段：仅校验整数转型（原实现无范围校验） ---
    def test_file_read_max_chars_non_int_raises(self):
        os.environ["KOCOR_FILE_READ_MAX_CHARS"] = "abc"
        with pytest.raises(ValueError):
            Config._load()

    def test_file_read_max_lines_non_int_raises(self):
        os.environ["KOCOR_FILE_READ_MAX_LINES"] = "abc"
        with pytest.raises(ValueError):
            Config._load()

    def test_file_search_max_results_non_int_raises(self):
        os.environ["KOCOR_FILE_SEARCH_MAX_RESULTS"] = "abc"
        with pytest.raises(ValueError):
            Config._load()

    def test_file_search_timeout_non_int_raises(self):
        os.environ["KOCOR_FILE_SEARCH_TIMEOUT"] = "abc"
        with pytest.raises(ValueError):
            Config._load()


