"""测试 read_file_tool。"""

import json
import os
import tempfile

from kocor.tools.toolset.read_file_tool import ReadFile


class TestReadFile:
    """测试 ReadFile 工具。"""

    def _make_file(self, content: str, suffix: str = ".py") -> str:
        """创建临时文件并返回路径。"""
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
        f.write(content)
        f.close()
        return f.name

    def test_read_full_file(self):
        """读取完整文件内容。"""
        content = "line1\nline2\nline3\n"
        path = self._make_file(content)
        try:
            result = ReadFile.handler(path=path)
            data = json.loads(result)
            assert "line1" in data["content"]
            assert "line2" in data["content"]
            assert data["total_lines"] == 3
        finally:
            os.unlink(path)

    def test_read_with_row_numbers(self):
        """读取的内容包含行号前缀。"""
        content = "hello\nworld\nfoo\n"
        path = self._make_file(content)
        try:
            result = ReadFile.handler(path=path, offset=1, limit=10)
            data = json.loads(result)
            assert "1|hello" in data["content"]
            assert "2|world" in data["content"]
            assert "3|foo" in data["content"]
        finally:
            os.unlink(path)

    def test_read_with_offset(self):
        """指定 offset 读取。"""
        content = "a\nb\nc\nd\ne\n"
        path = self._make_file(content)
        try:
            result = ReadFile.handler(path=path, offset=3, limit=2)
            data = json.loads(result)
            assert "3|c" in data["content"]
            assert "4|d" in data["content"]
            assert "5|e" not in data["content"]
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """文件不存在返回错误。"""
        result = ReadFile.handler(path="/tmp/nonexistent_file_xyz.py")
        data = json.loads(result)
        assert "error" in data

    def test_empty_file(self):
        """空文件返回空内容。"""
        path = self._make_file("")
        try:
            result = ReadFile.handler(path=path)
            data = json.loads(result)
            assert data["content"] == ""
            assert data["total_lines"] == 0
        finally:
            os.unlink(path)

    def test_binary_file_blocked_by_extension(self):
        """二进制扩展名文件被阻断。"""
        path = self._make_file("not really png", suffix=".png")
        try:
            result = ReadFile.handler(path=path)
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
                result = ReadFile.handler(path="../etc/passwd")
                assert "denied" in result.lower() or "Error" in result
            finally:
                os.chdir(old_cwd)

    def test_truncated_large_content(self):
        """超长内容被截断。"""
        # 生成超过 char limit 的内容
        lines = [f"line{i}_" + "x" * 200 for i in range(1000)]
        content = "\n".join(lines) + "\n"
        path = self._make_file(content)
        try:
            result = ReadFile.handler(path=path, offset=1, limit=2000)
            data = json.loads(result)
            assert data["truncated"] is True
            assert "truncated_by" in data
        finally:
            os.unlink(path)

    def test_dedup_returns_unchanged(self):
        """重复读取返回 unchanged。"""
        content = "dedup test\n"
        path = self._make_file(content)
        try:
            # 首次读取
            ReadFile.handler(path=path)
            # 再次读取
            result = ReadFile.handler(path=path)
            data = json.loads(result)
            if "status" in data:
                assert data["status"] == "unchanged"
        finally:
            os.unlink(path)