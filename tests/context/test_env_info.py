"""测试环境信息收集。"""

from __future__ import annotations

import os
from datetime import date

from kocor.context.system_prompt import SystemPromptBuilder


class TestEnvironmentInfo:
    """测试 _build_environment_info()。"""

    def test_returns_non_empty_string(self):
        result = SystemPromptBuilder._build_environment_info()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_date(self):
        result = SystemPromptBuilder._build_environment_info()
        assert date.today().isoformat() in result

    def test_contains_cwd(self):
        result = SystemPromptBuilder._build_environment_info()
        assert os.getcwd() in result

    def test_contains_os_info(self):
        result = SystemPromptBuilder._build_environment_info()
        assert "Windows" in result or "Linux" in result or "Darwin" in result

    def test_format_has_key_info(self):
        """环境信息应包含关键信息项。"""
        result = SystemPromptBuilder._build_environment_info()
        assert "当前日期:" in result
        assert "当前工作目录:" in result
        assert "操作系统:" in result

    def test_format_is_readable(self):
        result = SystemPromptBuilder._build_environment_info()
        # 应该有换行分隔的多行信息
        lines = result.split("\n")
        assert len(lines) >= 2
        for line in lines:
            assert len(line) > 0

    def test_does_not_contain_sensitive_info(self):
        """环境信息不应包含敏感数据。"""
        result = SystemPromptBuilder._build_environment_info()
        assert "API_KEY" not in result
        assert "PASSWORD" not in result.upper()
        assert "SECRET" not in result.upper()