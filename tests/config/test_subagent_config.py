"""测试子代理（subagent）工具配置项。

TDD Red：先写期望行为的测试，再在 Config 中实现字段使其通过。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from kocor.config import Config


class TestSubagentConfig:
    """测试 subagent_* 配置默认值与环境变量加载。"""

    _ENVS = [
        "KOCOR_SUBAGENT_ENABLED",
        "KOCOR_SUBAGENT_MAX_DEPTH",
        "KOCOR_SUBAGENT_MAX_CONCURRENT",
        "KOCOR_SUBAGENT_MAX_ITERATIONS",
        "KOCOR_SUBAGENT_MAX_SUMMARY_CHARS",
        "KOCOR_SUBAGENT_TIMEOUT",
        "KOCOR_SUBAGENT_AUTO_APPROVE",
        "KOCOR_SUBAGENT_BLOCKED_TOOLS",
    ]

    def setup_method(self):
        # 隔离 .env，避免开发环境配置回填污染默认值测试
        self._dotenv_patch = patch("kocor.config.load_dotenv")
        self._dotenv_patch.start()
        self._saved = {k: os.environ.pop(k, None) for k in self._ENVS}
        Config.reset()

    def teardown_method(self):
        self._dotenv_patch.stop()
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        Config.reset()

    # --- 默认值 ---
    def test_default_subagent_enabled(self):
        cfg = Config()
        assert cfg.subagent_enabled is True

    def test_default_subagent_max_depth(self):
        cfg = Config()
        assert cfg.subagent_max_depth == 1

    def test_default_subagent_max_concurrent(self):
        cfg = Config()
        assert cfg.subagent_max_concurrent == 3

    def test_default_subagent_max_iterations(self):
        cfg = Config()
        assert cfg.subagent_max_iterations == 15

    def test_default_subagent_max_summary_chars(self):
        cfg = Config()
        assert cfg.subagent_max_summary_chars == 8000

    def test_default_subagent_timeout(self):
        cfg = Config()
        assert cfg.subagent_timeout == 0

    def test_default_subagent_auto_approve(self):
        cfg = Config()
        assert cfg.subagent_auto_approve is False

    def test_default_subagent_blocked_tools(self):
        cfg = Config()
        assert cfg.subagent_blocked_tools == ("memory", "cronjob")

    # --- 环境变量加载 ---
    def test_load_subagent_enabled_from_env(self):
        os.environ["KOCOR_SUBAGENT_ENABLED"] = "false"
        cfg = Config._load()
        assert cfg.subagent_enabled is False

    def test_load_subagent_max_depth_from_env(self):
        os.environ["KOCOR_SUBAGENT_MAX_DEPTH"] = "2"
        cfg = Config._load()
        assert cfg.subagent_max_depth == 2

    def test_load_subagent_max_concurrent_from_env(self):
        os.environ["KOCOR_SUBAGENT_MAX_CONCURRENT"] = "5"
        cfg = Config._load()
        assert cfg.subagent_max_concurrent == 5

    def test_load_subagent_max_iterations_from_env(self):
        os.environ["KOCOR_SUBAGENT_MAX_ITERATIONS"] = "10"
        cfg = Config._load()
        assert cfg.subagent_max_iterations == 10

    def test_load_subagent_max_summary_chars_from_env(self):
        os.environ["KOCOR_SUBAGENT_MAX_SUMMARY_CHARS"] = "4000"
        cfg = Config._load()
        assert cfg.subagent_max_summary_chars == 4000

    def test_load_subagent_timeout_from_env(self):
        os.environ["KOCOR_SUBAGENT_TIMEOUT"] = "120"
        cfg = Config._load()
        assert cfg.subagent_timeout == 120

    def test_load_subagent_auto_approve_from_env(self):
        os.environ["KOCOR_SUBAGENT_AUTO_APPROVE"] = "true"
        cfg = Config._load()
        assert cfg.subagent_auto_approve is True

    # --- 数值范围校验 ---
    def test_max_depth_below_one_raises(self):
        os.environ["KOCOR_SUBAGENT_MAX_DEPTH"] = "0"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_max_concurrent_below_one_raises(self):
        os.environ["KOCOR_SUBAGENT_MAX_CONCURRENT"] = "0"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_max_iterations_below_one_raises(self):
        os.environ["KOCOR_SUBAGENT_MAX_ITERATIONS"] = "0"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_timeout_negative_raises(self):
        os.environ["KOCOR_SUBAGENT_TIMEOUT"] = "-1"
        try:
            Config._load()
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_summary_chars_zero_allowed(self):
        # 0 = 禁用截断，应被允许
        os.environ["KOCOR_SUBAGENT_MAX_SUMMARY_CHARS"] = "0"
        cfg = Config._load()
        assert cfg.subagent_max_summary_chars == 0

    def test_load_blocked_tools_from_env(self):
        os.environ["KOCOR_SUBAGENT_BLOCKED_TOOLS"] = "memory,cron"
        cfg = Config._load()
        assert cfg.subagent_blocked_tools == ("memory", "cron")

    def test_load_blocked_tools_single(self):
        os.environ["KOCOR_SUBAGENT_BLOCKED_TOOLS"] = "memory"
        cfg = Config._load()
        assert cfg.subagent_blocked_tools == ("memory",)

    def test_load_blocked_tools_empty_env_fallsback_default(self):
        """空字符串时回退默认值（env 不设置时为 None，不走 _coerce）。"""
        os.environ.pop("KOCOR_SUBAGENT_BLOCKED_TOOLS", None)
        # 需要重建 Config 以读取默认值
        cfg = Config()
        assert cfg.subagent_blocked_tools == ("memory", "cronjob")
