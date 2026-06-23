"""测试环境信息收集。"""

from __future__ import annotations

import os
from datetime import date

from kocor.context.env_info import build_environment_info


class TestEnvironmentInfo:
    """测试 build_environment_info()。"""

    def test_returns_non_empty_string(self):
        result = build_environment_info()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_date(self):
        result = build_environment_info()
        assert date.today().isoformat() in result

    def test_contains_cwd(self):
        result = build_environment_info()
        assert os.getcwd() in result

    def test_contains_os_info(self):
        result = build_environment_info()
        assert "Windows" in result or "Linux" in result or "Darwin" in result

    def test_contains_git_branch(self):
        """环境信息应包含 git 分支（如果在 git 仓库中）。"""
        result = build_environment_info()
        # 在 git 仓库中运行：应该有分支信息
        assert "Git 分支:" in result

    def test_format_is_readable(self):
        result = build_environment_info()
        # 应该有换行分隔的多行信息
        lines = result.split("\n")
        assert len(lines) >= 2
        for line in lines:
            assert len(line) > 0

    def test_does_not_contain_sensitive_info(self):
        """环境信息不应包含敏感数据。"""
        result = build_environment_info()
        assert "API_KEY" not in result
        assert "PASSWORD" not in result.upper()
        assert "SECRET" not in result.upper()