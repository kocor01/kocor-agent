"""RunPython 沙盒执行测试。"""

from unittest.mock import patch

import pytest

from kocor.tools.toolset.run_python import RunPython


class TestRunPythonExecution:
    """真实子进程执行测试。"""

    def test_simple_execution(self):
        result = RunPython.handler("print('hello')")
        assert "hello" in result

    def test_stderr_on_error(self):
        result = RunPython.handler("1/0")
        assert "ZeroDivisionError" in result

    def test_syntax_error(self):
        result = RunPython.handler("invalid syntax{{{")
        assert "SyntaxError" in result

    def test_blocked_module_os(self):
        result = RunPython.handler("import os; print(os.name)")
        assert "blocked" in result or "ImportError" in result

    def test_blocked_module_subprocess(self):
        result = RunPython.handler("import subprocess")
        assert "blocked" in result or "ImportError" in result

    def test_blocked_module_socket(self):
        result = RunPython.handler("import socket")
        assert "blocked" in result or "ImportError" in result

    def test_allowed_module_math(self):
        result = RunPython.handler("import math; print(math.pi)")
        assert "3.14" in result

    def test_stdlib_check_blocks_third_party(self):
        result = RunPython.handler("import requests")
        assert "ImportError" in result
        assert "requests" in result

    def test_env_sanitization(self):
        import os as os_module

        original = os_module.environ.copy()
        try:
            os_module.environ["TEST_API_KEY"] = "secret123"
            os_module.environ["TEST_SECRET_TOKEN"] = "mysecret"
            os_module.environ["TEST_SAFE_VAR"] = "safe"

            env = RunPython._build_env()
            assert "TEST_API_KEY" not in env
            assert "TEST_SECRET_TOKEN" not in env
            assert "TEST_SAFE_VAR" in env
        finally:
            os_module.environ.pop("TEST_API_KEY", None)
            os_module.environ.pop("TEST_SECRET_TOKEN", None)
            os_module.environ.pop("TEST_SAFE_VAR", None)


class TestRunPythonMocked:
    """使用 mock 的特定场景测试。"""

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_timeout(self, mock_run):
        from subprocess import TimeoutExpired

        mock_run.side_effect = TimeoutExpired(cmd="python", timeout=30)

        result = RunPython.handler("import time; time.sleep(10)")
        assert "timed out" in result

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_subprocess_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("python not found")

        result = RunPython.handler("print(1)")
        assert "FileNotFoundError" in result

    @patch("kocor.tools.toolset.run_python.subprocess.run")
    def test_pass_wrapper_not_raw_code(self, mock_run):
        mock_run.return_value = type("MockResult", (), {
            "returncode": 0, "stdout": "ok\n", "stderr": "",
        })()

        RunPython.handler("print(1)")
        args = mock_run.call_args.args
        # 第一个参数应该是 ["python", "-c", wrapper_code]
        assert args[0][0] == "python"
        assert args[0][1] == "-c"
        wrapper = args[0][2]
        assert "_safe_import" in wrapper
        assert "print(1)" in wrapper


class TestRunPythonStdlibCheck:
    """AST 静态检查测试。"""

    def test_stdlib_check_passes(self):
        assert RunPython._check_stdlib("import math") is None
        assert RunPython._check_stdlib("from datetime import datetime") is None
        assert RunPython._check_stdlib("import json, re") is None

    def test_stdlib_check_multiline(self):
        code = """import math
import json
print(math.pi)
"""
        assert RunPython._check_stdlib(code) is None

    def test_stdlib_check_blocks_third_party(self):
        err = RunPython._check_stdlib("import requests")
        assert err is not None
        assert "requests" in err

    def test_stdlib_check_blocks_from_import(self):
        err = RunPython._check_stdlib("from flask import Flask")
        assert err is not None
        assert "flask" in err

    def test_stdlib_check_syntax_error(self):
        err = RunPython._check_stdlib("invalid{{{")
        assert err is not None
        assert "SyntaxError" in err


class TestRunPythonBuildWrapper:
    """代码包装器测试。"""

    def test_wrapper_contains_import_blocker(self):
        wrapper = RunPython._build_wrapper("print(1)")
        assert "_safe_import" in wrapper
        assert "blocked =" in wrapper
        assert "os" in wrapper
        assert "subprocess" in wrapper

    def test_wrapper_preserves_user_code(self):
        wrapper = RunPython._build_wrapper("print('hello')")
        assert "print('hello')" in wrapper

    def test_build_env_has_pythonioencoding(self):
        env = RunPython._build_env()
        assert env.get("PYTHONIOENCODING") == "utf-8"
