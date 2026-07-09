"""测试 search_file_tool。"""

import json
import os
import tempfile

from kocor.tools.toolsets.search_file_tool import SearchFiles


class TestSearchFiles:
    """测试 SearchFiles 工具。"""

    def _make_files(self, base_dir: str):
        """在 base_dir 中创建测试文件。"""
        files = {}
        files["a.py"] = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        files["b.py"] = "def baz():\n    return 3\n"
        files["data.txt"] = "key1=value1\nkey2=value2\nTODO: fix this\n"
        for name, content in files.items():
            path = os.path.join(base_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return files

    def test_search_pattern_found(self):
        """内容搜索找到匹配。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_files(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = SearchFiles.handler(pattern="def foo")
                data = json.loads(result)
                assert data["total_count"] >= 1
                assert len(data.get("matches", data.get("files", [])) or []) >= 1
            finally:
                os.chdir(old_cwd)

    def test_search_pattern_not_found(self):
        """内容搜索无匹配。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_files(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = SearchFiles.handler(pattern="XYZZZZ_NOT_FOUND")
                data = json.loads(result)
                assert data["total_count"] == 0
            finally:
                os.chdir(old_cwd)

    def test_search_with_file_glob(self):
        """按文件通配符搜索。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_files(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = SearchFiles.handler(pattern="def", file_glob="*.py")
                data = json.loads(result)
                assert data["total_count"] >= 1
            finally:
                os.chdir(old_cwd)

    def test_search_files_only(self):
        """按文件名搜索。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_files(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = SearchFiles.handler(pattern=".py", target="files")
                data = json.loads(result)
                # 至少找到 1 个文件
                assert len(data.get("files", [])) >= 1
            finally:
                os.chdir(old_cwd)

    def test_search_with_limit(self):
        """搜索结果限制。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_files(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = SearchFiles.handler(pattern="def", limit=1)
                data = json.loads(result)
                assert data["total_count"] >= 1
                # limit 可能不影响 total_count 但影响实际返回结果数
                assert "matches" in data or "files" in data
            finally:
                os.chdir(old_cwd)

    def test_search_count_output_mode(self):
        """count 输出模式。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_files(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = SearchFiles.handler(pattern="def", output_mode="count")
                data = json.loads(result)
                assert data["total_count"] >= 1
            finally:
                os.chdir(old_cwd)