"""测试项目指令加载。"""

from __future__ import annotations

import os
import tempfile

from kocor.context.system_prompt import SystemPromptBuilder


class TestProjectInstructions:
    """测试 _load_project_instructions()。"""

    def test_file_not_found_returns_empty(self):
        """文件不存在应返回空字符串。"""
        result = SystemPromptBuilder._load_project_instructions("/tmp/nonexistent_file_KOCOR_test.md")
        assert result == ""

    def test_empty_file_returns_empty(self):
        """空文件应返回空字符串。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write("")
            path = f.name

        try:
            result = SystemPromptBuilder._load_project_instructions(path)
            assert result == ""
        finally:
            os.unlink(path)

    def test_reads_file_content(self):
        """应正确读取文件内容。"""
        content = "## 项目规范\n\n- 使用 Python 3.12\n- 遵循 PEP 8"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = SystemPromptBuilder._load_project_instructions(path)
            assert "## 项目规范" in result
            assert "Python 3.12" in result
        finally:
            os.unlink(path)

    def test_includes_section_header(self):
        """返回内容应包含 ## 项目指令 标题。"""
        content = "一些指令"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(content)
            path = f.name

        try:
            result = SystemPromptBuilder._load_project_instructions(path)
            assert "项目指令" in result
        finally:
            os.unlink(path)

    def test_default_path_does_not_exist(self):
        """默认路径 KOCOR.md 不存在时应正常返回空字符串。"""
        # 确保当前目录没有 KOCOR.md
        if os.path.exists("KOCOR.md"):
            os.rename("KOCOR.md", "KOCOR.md.bak")
            cleaned = True
        else:
            cleaned = False

        try:
            result = SystemPromptBuilder._load_project_instructions()
            assert result == ""
        finally:
            if cleaned:
                os.rename("KOCOR.md.bak", "KOCOR.md")