"""测试文档提取模块。"""

import json
import os
import tempfile

from kocor.tools.toolset.read_extract import (
    extract_document_text,
    is_extractable_document,
)


class TestIsExtractableDocument:
    """测试文档类型判断。"""

    def test_ipynb_returns_true(self):
        assert is_extractable_document("notebook.ipynb") is True

    def test_py_returns_false(self):
        assert is_extractable_document("script.py") is False

    def test_txt_returns_false(self):
        assert is_extractable_document("readme.txt") is False

    def test_no_extension_returns_false(self):
        assert is_extractable_document("Makefile") is False


class TestExtractNotebook:
    """测试 Jupyter Notebook 提取。"""

    def _make_notebook(self, cells: list[dict]) -> str:
        """创建临时 .ipynb 文件。"""
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": cells,
        }
        f = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w", encoding="utf-8")
        json.dump(nb, f)
        f.close()
        return f.name

    def test_code_cell(self):
        """代码单元格被提取。"""
        cells = [
            {
                "cell_type": "code",
                "source": ["print('hello')\n", "print('world')\n"],
                "metadata": {},
            }
        ]
        path = self._make_notebook(cells)
        try:
            result = extract_document_text(path)
            assert "print('hello')" in result
            assert "print('world')" in result
            assert "[code]" in result.lower()
        finally:
            os.unlink(path)

    def test_markdown_cell(self):
        """Markdown 单元格被提取。"""
        cells = [
            {
                "cell_type": "markdown",
                "source": ["# Title\n", "Some *text*\n"],
                "metadata": {},
            }
        ]
        path = self._make_notebook(cells)
        try:
            result = extract_document_text(path)
            assert "# Title" in result
            assert "Some *text*" in result
            assert "[markdown]" in result.lower()
        finally:
            os.unlink(path)

    def test_mixed_cells(self):
        """混合单元格按顺序提取。"""
        cells = [
            {
                "cell_type": "markdown",
                "source": ["# Introduction\n"],
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": ["import os\n"],
                "metadata": {},
                "outputs": [],
            },
        ]
        path = self._make_notebook(cells)
        try:
            result = extract_document_text(path)
            assert "# Introduction" in result
            assert "import os" in result
            # 顺序检查
            intro_idx = result.index("# Introduction")
            import_idx = result.index("import os")
            assert intro_idx < import_idx
        finally:
            os.unlink(path)

    def test_code_with_outputs(self):
        """代码单元格的输出被提取。"""
        cells = [
            {
                "cell_type": "code",
                "source": ["print(42)\n"],
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "stream",
                        "text": ["42\n"],
                        "name": "stdout",
                    }
                ],
            }
        ]
        path = self._make_notebook(cells)
        try:
            result = extract_document_text(path)
            assert "print(42)" in result
            assert "42" in result
        finally:
            os.unlink(path)

    def test_raw_cell(self):
        """Raw 单元格被提取。"""
        cells = [
            {
                "cell_type": "raw",
                "source": ["raw content\n"],
                "metadata": {},
            }
        ]
        path = self._make_notebook(cells)
        try:
            result = extract_document_text(path)
            assert "raw content" in result
        finally:
            os.unlink(path)

    def test_empty_notebook(self):
        """空笔记本提取为空。"""
        cells = []
        path = self._make_notebook(cells)
        try:
            result = extract_document_text(path)
            assert result == ""
        finally:
            os.unlink(path)

    def test_invalid_json(self):
        """无效 JSON 文件抛出异常。"""
        f = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w")
        f.write("not json content")
        f.close()
        try:
            import pytest
            from kocor.tools.toolset.read_extract import ExtractionError
            with pytest.raises(ExtractionError):
                extract_document_text(f.name)
        finally:
            os.unlink(f.name)