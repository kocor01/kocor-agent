"""FileAccessGuard 测试。"""

import os
import tempfile
import pytest
from kocor.harness.file_guard import FileAccessGuard


class TestFileAccessGuard:
    def test_no_restriction_allows_any_read(self):
        guard = FileAccessGuard()
        path = guard.check_read("/any/path.txt")
        assert path == os.path.abspath("/any/path.txt")

    def test_read_within_allowed_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = FileAccessGuard(allowed_dir=tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            path = guard.check_read(test_file)
            assert path == os.path.abspath(test_file)

    def test_read_outside_allowed_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = FileAccessGuard(allowed_dir=tmpdir)
            with pytest.raises(PermissionError, match="拒绝"):
                guard.check_read("/etc/passwd")

    def test_write_outside_allowed_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = FileAccessGuard(allowed_dir=tmpdir)
            with pytest.raises(PermissionError, match="拒绝"):
                guard.check_write("/etc/passwd")

    def test_write_env_file_raises(self):
        guard = FileAccessGuard()
        with pytest.raises(PermissionError, match="敏感文件"):
            guard.check_write("/some/dir/.env")

    def test_write_env_local_file_raises(self):
        guard = FileAccessGuard()
        with pytest.raises(PermissionError, match="敏感文件"):
            guard.check_write("/some/dir/.env.local")

    def test_write_regular_file_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = FileAccessGuard(allowed_dir=tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            path = guard.check_write(test_file)
            assert path == os.path.abspath(test_file)

    def test_no_restriction_allows_write(self):
        guard = FileAccessGuard()
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            path = guard.check_write(test_file)
            assert path == os.path.abspath(test_file)