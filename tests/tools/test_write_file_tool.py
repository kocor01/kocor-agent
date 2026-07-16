"""测试 write_file_tool。"""

import json
import os
import tempfile

from kocor.tools.toolsets.file.file_state import FileStateTracker
from kocor.tools.toolsets.write_file_tool import WriteFile
from tests.tools.conftest import chdir_cm


class TestWriteFile:
    """测试 WriteFile 工具。"""

    def setup_method(self):
        self.tracker = FileStateTracker()

    def test_write_new_file(self):
        """写入新文件。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "test.py")
            result = WriteFile.handler(file_state=self.tracker, path=path, content="print('hello')\n")
            data = json.loads(result)
            assert "bytes_written" in data
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                assert f.read() == "print('hello')\n"

    def test_write_creates_dirs(self):
        """写入时自动创建父目录。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "sub", "nested", "test.py")
            result = WriteFile.handler(file_state=self.tracker, path=path, content="nested\n")
            data = json.loads(result)
            assert "bytes_written" in data
            assert os.path.exists(path)

    def test_overwrite_existing_file(self):
        """覆盖已存在的文件。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "test.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("old content\n")
            WriteFile.handler(file_state=self.tracker, path=path, content="new content\n")
            with open(path, encoding="utf-8") as f:
                assert f.read() == "new content\n"

    def test_write_empty_content(self):
        """写入空内容。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "empty.txt")
            result = WriteFile.handler(file_state=self.tracker, path=path, content="")
            data = json.loads(result)
            assert data["bytes_written"] == 0
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                assert f.read() == ""

    def test_path_traversal_rejected(self):
        """路径遍历被拒绝。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            result = WriteFile.handler(file_state=self.tracker, path="../outside.txt", content="hack")
            assert "denied" in result.lower() or "Error" in result

    def test_sensitive_path_rejected(self):
        """敏感系统路径被拒绝。"""
        result = WriteFile.handler(file_state=self.tracker, path="/etc/passwd", content="hack")
        data = json.loads(result)
        assert "error" in data

    def test_env_file_rejected(self):
        """.env 文件被拒绝写入。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, ".env")
            result = WriteFile.handler(file_state=self.tracker, path=path, content="SECRET=xxx")
            data = json.loads(result)
            assert "error" in data

    def test_internal_tool_content_rejected(self):
        """内部工具显示文本被拒绝写入。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "test.py")
            # 模拟 read_file 的行号内容
            content = "1|import os\n2|import sys\n3|\n4|def main():\n5|    pass"
            result = WriteFile.handler(file_state=self.tracker, path=path, content=content)
            data = json.loads(result)
            assert "error" in data

    def test_preserves_crlf(self):
        """保留 CRLF 行尾。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "crlf.txt")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("line1\r\nline2\r\n")
            WriteFile.handler(file_state=self.tracker, path=path, content="line3\r\nline4\r\n")
            with open(path, "rb") as f:
                content = f.read()
            assert b"\r\n" in content
            # 内容是 \r\n 行尾
            assert content.count(b"\r\n") == 2

    def test_write_utf8_with_bom(self):
        """保留 UTF-8 BOM。"""
        with tempfile.TemporaryDirectory() as tmpdir, chdir_cm(tmpdir):
            path = os.path.join(tmpdir, "bom.txt")
            # 写入带 BOM 的文件
            raw = "﻿Hello\nWorld\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(raw)
            WriteFile.handler(file_state=self.tracker, path=path, content="New\nContent\n")
            with open(path, "rb") as f:
                content = f.read()
            # BOM 应保留
            assert content[:3] == b"\xef\xbb\xbf"
            assert b"New" in content
            assert b"Content" in content
