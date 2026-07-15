"""测试 read_file_tool。"""

import json
import os
import tempfile

from kocor.tools.toolsets.read_file_tool import ReadFile
from kocor.tools.toolsets.file.file_state import FileStateTracker


class TestReadFile:
    """测试 ReadFile 工具。"""

    def setup_method(self):
        self.tracker = FileStateTracker()

    def _make_file(self, content: str, suffix: str = ".py", monkeypatch=None) -> str:
        """创建临时文件并返回路径。

        read_file 工具以 os.getcwd() 为允许目录，临时文件位于系统 temp
        目录（工作目录之外）会被路径越界检查拒绝。传入 monkeypatch 时
        chdir 到文件所在目录，使绝对路径落在允许目录内。
        """
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
        f.write(content)
        f.close()
        if monkeypatch is not None:
            monkeypatch.chdir(os.path.dirname(f.name))
        return f.name

    def test_read_full_file(self, monkeypatch):
        """读取完整文件内容。"""
        content = "line1\nline2\nline3\n"
        path = self._make_file(content, monkeypatch=monkeypatch)
        try:
            result = ReadFile.handler(file_state=self.tracker, path=path)
            data = json.loads(result)
            assert "line1" in data["content"]
            assert "line2" in data["content"]
            assert data["total_lines"] == 3
        finally:
            os.unlink(path)

    def test_read_with_row_numbers(self, monkeypatch):
        """读取的内容包含行号前缀。"""
        content = "hello\nworld\nfoo\n"
        path = self._make_file(content, monkeypatch=monkeypatch)
        try:
            result = ReadFile.handler(file_state=self.tracker, path=path, offset=1, limit=10)
            data = json.loads(result)
            assert "1|hello" in data["content"]
            assert "2|world" in data["content"]
            assert "3|foo" in data["content"]
        finally:
            os.unlink(path)

    def test_read_with_offset(self, monkeypatch):
        """指定 offset 读取。"""
        content = "a\nb\nc\nd\ne\n"
        path = self._make_file(content, monkeypatch=monkeypatch)
        try:
            result = ReadFile.handler(file_state=self.tracker, path=path, offset=3, limit=2)
            data = json.loads(result)
            assert "3|c" in data["content"]
            assert "4|d" in data["content"]
            assert "5|e" not in data["content"]
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """文件不存在返回错误。"""
        result = ReadFile.handler(file_state=self.tracker, path="/tmp/nonexistent_file_xyz.py")
        data = json.loads(result)
        assert "error" in data

    def test_empty_file(self, monkeypatch):
        """空文件返回空内容。"""
        path = self._make_file("", monkeypatch=monkeypatch)
        try:
            result = ReadFile.handler(file_state=self.tracker, path=path)
            data = json.loads(result)
            assert data["content"] == ""
            assert data["total_lines"] == 0
        finally:
            os.unlink(path)

    def test_binary_file_blocked_by_extension(self, monkeypatch):
        """二进制扩展名文件被阻断。"""
        path = self._make_file("not really png", suffix=".png", monkeypatch=monkeypatch)
        try:
            result = ReadFile.handler(file_state=self.tracker, path=path)
            data = json.loads(result)
            assert "error" in data
            assert "binary" in data["error"].lower() or "Cannot" in data["error"]
        finally:
            os.unlink(path)

    def test_path_traversal_rejected(self):
        """路径遍历被拒绝。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = ReadFile.handler(file_state=self.tracker, path="../etc/passwd")
                assert "denied" in result.lower() or "Error" in result
            finally:
                os.chdir(old_cwd)

    def test_truncated_large_content(self, monkeypatch):
        """超长内容被截断。"""
        # 生成超过 char limit 的内容
        lines = [f"line{i}_" + "x" * 200 for i in range(1000)]
        content = "\n".join(lines) + "\n"
        path = self._make_file(content, monkeypatch=monkeypatch)
        try:
            result = ReadFile.handler(file_state=self.tracker, path=path, offset=1, limit=2000)
            data = json.loads(result)
            assert data["truncated"] is True
            assert "truncated_by" in data
        finally:
            os.unlink(path)

    def test_dedup_returns_unchanged(self, monkeypatch):
        """重复读取返回 unchanged。"""
        content = "dedup test\n"
        path = self._make_file(content, monkeypatch=monkeypatch)
        try:
            # 首次读取
            ReadFile.handler(file_state=self.tracker, path=path)
            # 再次读取
            result = ReadFile.handler(file_state=self.tracker, path=path)
            data = json.loads(result)
            if "status" in data:
                assert data["status"] == "unchanged"
        finally:
            os.unlink(path)
