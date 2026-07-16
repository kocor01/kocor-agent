"""Config 热加载功能测试。"""

from __future__ import annotations

import os

from kocor.config import Config


def _dummy_hook(cfg):
    """测试用回调，记录被调用。"""
    _dummy_hook.called = True
    _dummy_hook.config = cfg


class TestHotReload:
    def test_register_reload_hook(self):
        Config.reset()
        _dummy_hook.called = False
        Config.register_reload_hook(_dummy_hook)
        assert _dummy_hook in Config._reload_hooks

    def test_notify_reload_calls_hooks(self):
        Config.reset()
        _dummy_hook.called = False
        Config.register_reload_hook(_dummy_hook)
        Config._notify_reload()
        assert _dummy_hook.called

    def test_notify_reload_loads_fresh_config(self):
        """重载通知应加载最新环境变量。"""
        Config.reset()
        os.environ["KOCOR_MAX_ITERATIONS"] = "10"
        Config.load()

        os.environ["KOCOR_MAX_ITERATIONS"] = "30"

        received = []

        def capture(cfg):
            received.append(cfg.max_iterations)

        Config.register_reload_hook(capture)
        Config._notify_reload()
        assert received[0] == 30
        del os.environ["KOCOR_MAX_ITERATIONS"]

    def test_enable_hot_reload_platform_check(self):
        """enable_hot_reload 应返回平台是否支持。"""
        Config.reset()
        result = Config.enable_hot_reload()
        if os.name == "nt":
            assert not result  # Windows 不支持 SIGHUP
        else:
            assert result

    def test_hot_reload_does_not_crash_on_bad_hook(self):
        """错误的回调不应导致其他回调失败。"""
        Config.reset()

        def bad_hook(cfg):
            raise RuntimeError("hook failed")

        good_hook_called = [False]

        def good_hook(cfg):
            good_hook_called[0] = True

        Config.register_reload_hook(bad_hook)
        Config.register_reload_hook(good_hook)

        # 不应抛出异常
        Config._notify_reload()
        assert good_hook_called[0]

    def test_reload_hook_receives_new_config(self):
        """回调应收到新 Config 实例。"""
        Config.reset()
        os.environ["KOCOR_MAX_ITERATIONS"] = "50"
        received = []
        Config.register_reload_hook(lambda cfg: received.append(cfg))
        Config._notify_reload()
        assert len(received) == 1
        assert received[0].max_iterations == 50
        del os.environ["KOCOR_MAX_ITERATIONS"]