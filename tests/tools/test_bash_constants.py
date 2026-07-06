"""bash/constants.py 单元测试：shell 查找、路径常量、环境构建。"""

import os
import platform
from unittest.mock import patch

import pytest

from kocor.tools.toolset.bash.constants import (
    IS_WINDOWS,
    _find_bash,
    _make_run_env,
    _resolve_safe_cwd,
    _quote_cwd_for_cd,
)


class TestIsWindows:
    """平台检测测试。"""

    def test_is_windows_matches_platform(self):
        assert IS_WINDOWS == (platform.system() == "Windows")


class TestFindBash:
    """跨平台 bash 查找测试。"""

    @patch("kocor.tools.toolset.bash.constants.shutil.which")
    @patch("kocor.tools.toolset.bash.constants.IS_WINDOWS", False)
    def test_posix_uses_shutil(self, mock_which):
        mock_which.return_value = "/usr/bin/bash"
        assert _find_bash() == "/usr/bin/bash"
        mock_which.assert_called_once_with("bash")

    @patch("kocor.tools.toolset.bash.constants.shutil.which")
    @patch("kocor.tools.toolset.bash.constants.IS_WINDOWS", False)
    def test_posix_fallback_to_bin_bash(self, mock_which):
        mock_which.return_value = None
        result = _find_bash()
        assert result == "/bin/bash"

    @patch("kocor.tools.toolset.bash.constants.IS_WINDOWS", True)
    def test_windows_uses_env_var(self):
        with patch.dict(os.environ, {"KOCOR_GIT_BASH_PATH": "C:\\tools\\bash.exe"}, clear=True):
            with patch("os.path.isfile", return_value=True):
                assert _find_bash() == "C:\\tools\\bash.exe"

    @patch("kocor.tools.toolset.bash.constants.IS_WINDOWS", True)
    def test_windows_uses_programfiles(self):
        with patch.dict(os.environ, {"ProgramFiles": "C:\\Program Files"}, clear=True):
            with patch("os.path.isfile", side_effect=[False, True, False]):
                result = _find_bash()
                assert "Git\\bin\\bash.exe" in result

    @patch("kocor.tools.toolset.bash.constants.IS_WINDOWS", True)
    def test_windows_raises_when_not_found(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.isfile", return_value=False):
                with patch("shutil.which", return_value=None):
                    with pytest.raises(RuntimeError, match="bash not found"):
                        _find_bash()


class TestMakeRunEnv:
    """子进程环境构建测试。"""

    def test_contains_pythonunbuffered(self):
        env = _make_run_env()
        assert env.get("PYTHONUNBUFFERED") == "1"

    def test_strips_sensitive_keys(self):
        with patch.dict(os.environ, {
            "SAFE_VAR": "hello",
            "MY_API_KEY": "secret123",
            "MY_SECRET_TOKEN": "tokensecret",
            "MY_SECRET_VALUE": "mysecret",
        }, clear=True):
            env = _make_run_env()
            assert env.get("SAFE_VAR") == "hello"
            assert "MY_API_KEY" not in env
            assert "MY_SECRET_TOKEN" not in env
            assert "MY_SECRET_VALUE" in env  # 不以 _API_KEY/_SECRET/_TOKEN 结尾

    def test_strips_sk_like_keys(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test1234567890",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }, clear=True):
            env = _make_run_env()
            assert "OPENAI_API_KEY" not in env
            assert "ANTHROPIC_API_KEY" not in env

    def test_keeps_normal_env(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin", "HOME": "/home/user"}, clear=True):
            env = _make_run_env()
            assert env.get("PATH") == "/usr/bin:/bin"
            assert env.get("HOME") == "/home/user"


class TestResolveSafeCwd:
    """安全 CWD 解析测试。"""

    def test_returns_cwd_when_exists(self, tmp_path):
        assert _resolve_safe_cwd(str(tmp_path)) == str(tmp_path)

    def test_falls_back_to_parent(self, tmp_path):
        missing = str(tmp_path / "nonexistent" / "deeper")
        assert _resolve_safe_cwd(missing) == str(tmp_path)

    def test_falls_back_to_tempdir(self):
        import tempfile
        result = _resolve_safe_cwd("")
        assert result == tempfile.gettempdir()


class TestQuoteCwdForCd:
    """cd 目标路径引号测试。"""

    def test_tilde_is_preserved(self):
        assert _quote_cwd_for_cd("~") == "~"

    def test_tilde_slash_becomes_home(self):
        assert _quote_cwd_for_cd("~/") == "$HOME"

    def test_tilde_path_uses_home(self):
        result = _quote_cwd_for_cd("~/projects/my app")
        assert "$HOME" in result
        assert "my app" in result

    def test_absolute_path_is_quoted(self):
        result = _quote_cwd_for_cd("/home/user/project")
        assert result == "/home/user/project"

    def test_path_with_spaces_is_quoted(self):
        result = _quote_cwd_for_cd("/home/user/my project")
        assert "my project" in result or "'my project'" in result