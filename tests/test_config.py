"""测试配置加载"""

import json
import os
from unittest.mock import mock_open, patch

import pytest

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


class TestMCPConfigValidation:
    """测试 MCP 配置文件校验"""

    def test_validate_valid_json(self):
        from kocor.config import _validate_mcp_config_json

        data = json.dumps({"mcpServers": {"fs": {"command": "npx"}}})
        with patch("builtins.open", mock_open(read_data=data)):
            _validate_mcp_config_json("valid.json")  # 不应抛出

    def test_validate_invalid_json_raises(self):
        from kocor.config import _validate_mcp_config_json

        with patch("builtins.open", mock_open(read_data="not json")):
            with pytest.raises(ValueError, match="JSON"):
                _validate_mcp_config_json("bad.json")

    def test_validate_missing_servers_field_raises(self):
        from kocor.config import _validate_mcp_config_json

        data = json.dumps({"other": "value"})
        with patch("builtins.open", mock_open(read_data=data)):
            with pytest.raises(ValueError, match="mcpServers"):
                _validate_mcp_config_json("no_servers.json")

    def test_validate_servers_not_dict_raises(self):
        from kocor.config import _validate_mcp_config_json

        data = json.dumps({"mcpServers": "not_a_dict"})
        with patch("builtins.open", mock_open(read_data=data)):
            with pytest.raises(ValueError, match="mcpServers"):
                _validate_mcp_config_json("bad_servers.json")

    def test_env_var_invalid_path_raises(self):
        os.environ["KOCOR_MCP_CONFIG"] = "/nonexistent/path.json"
        os.environ["KOCOR_PROVIDER"] = "openai"
        try:
            load_config()
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "不存在" in str(e) or "not exist" in str(e)
        finally:
            os.environ.pop("KOCOR_MCP_CONFIG", None)
            os.environ.pop("KOCOR_PROVIDER", None)

    def test_env_var_invalid_json_raises(self):
        os.environ["KOCOR_MCP_CONFIG"] = "bad_mcp_config.json"
        os.environ["KOCOR_PROVIDER"] = "openai"
        try:
            with patch("kocor.config.os.path.exists", return_value=True), \
                 patch("builtins.open", mock_open(read_data="not json")):
                load_config()
                assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "JSON" in str(e)
        finally:
            os.environ.pop("KOCOR_MCP_CONFIG", None)
            os.environ.pop("KOCOR_PROVIDER", None)
