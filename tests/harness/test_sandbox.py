"""Sandbox 执行环境测试。"""

import pytest
from kocor.harness.sandbox import Sandbox


class TestSandbox:
    def test_simple_execution(self):
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("print('hello')")
        assert "hello" in result.stdout
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_stderr_on_error(self):
        """错误会被捕获到 stderr。"""
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("1/0")
        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    def test_syntax_error(self):
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("invalid syntax{{{")
        assert result.exit_code != 0
        assert result.stderr

    def test_timeout(self):
        sandbox = Sandbox(timeout=1)
        result = sandbox.run("import time; time.sleep(10)")
        assert result.timed_out is True

    def test_blocked_module_os(self):
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("import os; print(os.name)")
        assert "blocked" in result.stderr or "ImportError" in result.stderr
        assert result.exit_code != 0

    def test_blocked_module_subprocess(self):
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("import subprocess")
        assert "blocked" in result.stderr or "ImportError" in result.stderr
        assert result.exit_code != 0

    def test_allowed_module_math(self):
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("import math; print(math.pi)")
        assert result.exit_code == 0
        assert "3.14" in result.stdout

    def test_network_disabled_by_default(self):
        sandbox = Sandbox(timeout=5, network_access=False)
        result = sandbox.run("import urllib.request")
        # urllib 默认被拦截
        assert "blocked" in result.stderr or result.exit_code != 0

    def test_env_sanitization(self):
        """敏感环境变量应在沙箱中被移除。"""
        import os
        original = os.environ.copy()
        try:
            os.environ["TEST_API_KEY"] = "secret123"
            os.environ["TEST_SECRET"] = "mysecret"
            os.environ["TEST_SAFE_VAR"] = "safe"

            sandbox = Sandbox(timeout=10)
            env = sandbox._build_env()

            # API_KEY 和 SECRET 变量应被移除
            assert "TEST_API_KEY" not in env
            assert "TEST_SECRET" not in env
            # 非敏感变量应保留
        finally:
            # 清理测试环境变量
            os.environ.pop("TEST_API_KEY", None)
            os.environ.pop("TEST_SECRET", None)
            os.environ.pop("TEST_SAFE_VAR", None)

    def test_allow_additional_modules(self):
        """设置 allowed_modules 默认不影响 blocked 列表。"""
        sandbox = Sandbox(timeout=10)
        # 默认情况下，即使 allowed_modules 为空，os 仍然被拦截
        result = sandbox.run("import os; print(os.name)")
        assert result.exit_code != 0

    def test_custom_blocked_modules(self):
        sandbox = Sandbox(timeout=10, blocked_modules={"json"})
        result = sandbox.run("import json")
        assert "blocked" in result.stderr or "ImportError" in result.stderr
        assert result.exit_code != 0

    def test_result_attributes(self):
        sandbox = Sandbox(timeout=10)
        result = sandbox.run("print(42)")
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "exit_code")
        assert hasattr(result, "timed_out")


class TestSandboxResult:
    def test_from_subprocess(self):
        from kocor.harness.sandbox import SandboxResult
        r = SandboxResult(stdout="hello", stderr="", exit_code=0, timed_out=False)
        assert r.stdout == "hello"
        assert r.success is True

    def test_failure(self):
        from kocor.harness.sandbox import SandboxResult
        r = SandboxResult(stdout="", stderr="error", exit_code=1, timed_out=False)
        assert r.success is False