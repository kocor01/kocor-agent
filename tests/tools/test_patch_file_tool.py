"""测试 patch_file_tool。"""

import json
import os
import tempfile

from kocor.tools.toolsets.patch_file_tool import PatchFile
from kocor.tools.toolsets.file.file_state import FileStateTracker


class TestPatchFile:
    """测试 PatchFile 工具。"""

    def setup_method(self):
        self.tracker = FileStateTracker()

    def _make_file(self, content: str, suffix: str = ".py") -> str:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
        f.write(content)
        f.close()
        return f.name

    def test_exact_replace(self):
        """精确匹配替换。"""
        content = "def foo():\n    return 1\n"
        path = self._make_file(content)
        try:
            result = PatchFile.handler(file_state=self.tracker, path=path, old_string="def foo():", new_string="def bar():")
            data = json.loads(result)
            assert data["success"] is True
            assert "diff" in data
            with open(path, encoding="utf-8") as f:
                assert "def bar():" in f.read()
        finally:
            os.unlink(path)

    def test_fuzzy_replace(self):
        """模糊匹配替换。"""
        content = "    if x:\n        return 1\n"
        path = self._make_file(content)
        try:
            result = PatchFile.handler(file_state=self.tracker, path=path, old_string="if x:\n    return 1", new_string="if x:\n    return 2")
            data = json.loads(result)
            assert data["success"] is True
            with open(path, encoding="utf-8") as f:
                assert "return 2" in f.read()
        finally:
            os.unlink(path)

    def test_replace_all(self):
        """替换全部。"""
        content = "x = 1\nx = 2\nx = 3\n"
        path = self._make_file(content)
        try:
            result = PatchFile.handler(file_state=self.tracker, path=path, old_string="x = ", new_string="y = ", replace_all=True)
            data = json.loads(result)
            assert data["success"] is True
            with open(path, encoding="utf-8") as f:
                assert "y = 3" in f.read()
        finally:
            os.unlink(path)

    def test_sensitive_path_rejected(self):
        """敏感路径被拒绝。"""
        result = PatchFile.handler(file_state=self.tracker, path="/etc/passwd", old_string="root", new_string="admin")
        data = json.loads(result)
        assert "error" in data

    def test_no_match_returns_error(self):
        """无匹配返回错误，文件不变。"""
        content = "original content\n"
        path = self._make_file(content)
        try:
            result = PatchFile.handler(file_state=self.tracker, path=path, old_string="nonexistent", new_string="replacement")
            data = json.loads(result)
            assert data["success"] is False
            assert "error" in data
            with open(path, encoding="utf-8") as f:
                assert f.read() == content
        finally:
            os.unlink(path)

    def test_different_line_endings(self):
        """匹配不因行尾差异失败。"""
        content = "def foo():\n    return 1\n"
        path = self._make_file(content)
        try:
            # old_string 行尾差异（\r\n vs \n）
            result = PatchFile.handler(file_state=self.tracker, path=path, old_string="def foo():\r\n    return 1", new_string="def bar():\r\n    return 2")
            data = json.loads(result)
            assert data["success"] is True
            with open(path, encoding="utf-8") as f:
                assert "def bar():" in f.read()
        finally:
            os.unlink(path)

    def test_env_file_rejected(self):
        """.env 文件被拒绝。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".env")
            with open(path, "w") as f:
                f.write("KEY=value\n")
            result = PatchFile.handler(file_state=self.tracker, path=path, old_string="KEY=value", new_string="KEY=newvalue")
            data = json.loads(result)
            assert "error" in data

    def test_internal_tool_content_rejected(self):
        """行号前缀内容被拒绝写入。"""
        content = "def foo():\n    pass\n"
        path = self._make_file(content)
        try:
            result = PatchFile.handler(
                path=path,
                old_string="def foo():",
                new_string="1|def foo():\n2|    pass",
            )
            data = json.loads(result)
            assert "error" in data
        finally:
            os.unlink(path)