"""HarnessConfig 测试。"""

import dataclasses

from kocor.harness.config import HarnessConfig
from kocor.tools.permission import PermissionManager


def _update_from_dict(config, data):
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)


class TestHarnessConfig:
    def test_default_values(self):
        config = HarnessConfig()
        assert config.max_iterations == 20
        assert config.permission_policy == PermissionManager.POLICY_DEFAULT
        assert config.context_max_tokens == 200_000

    def test_custom_values(self):
        config = HarnessConfig(
            max_iterations=10,
            permission_policy=PermissionManager.POLICY_STRICT,
        )
        assert config.max_iterations == 10
        assert config.permission_policy == PermissionManager.POLICY_STRICT

    def test_context_thresholds(self):
        config = HarnessConfig()
        assert config.context_summary_threshold == 0.70
        assert config.context_truncate_threshold == 0.90
        assert config.preserve_last_rounds == 3
        assert config.preserve_first_rounds == 1

    def test_update_from_dict(self):
        config = HarnessConfig()
        _update_from_dict(config, {
            "max_iterations": 5,
        })
        assert config.max_iterations == 5

    def test_update_from_dict_ignores_unknown(self):
        config = HarnessConfig()
        _update_from_dict(config, {"unknown_field": 42})
        # 不应抛出异常，应静默忽略

    def test_to_dict(self):
        config = HarnessConfig(max_iterations=15)
        d = dataclasses.asdict(config)
        assert d["max_iterations"] == 15
        assert d["permission_policy"] == PermissionManager.POLICY_DEFAULT