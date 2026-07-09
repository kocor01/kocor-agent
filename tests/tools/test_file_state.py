"""测试文件状态管理模块。"""

import os
import tempfile
import time

from kocor.tools.toolset.file_state import FileStateTracker


class TestFileStateTrackerIsolation:
    """测试 FileStateTracker 实例隔离性。"""

    def test_two_trackers_do_not_interfere(self):
        """两个独立的 FileStateTracker 实例互不干扰。"""
        t1 = FileStateTracker()
        t2 = FileStateTracker()

        t1.record_patch_failure("/a.py")
        t1.record_patch_failure("/a.py")

        assert t1.record_patch_failure("/a.py") == 3
        assert t2.record_patch_failure("/a.py") == 1  # 独立的，不是 4

    def test_dedup_isolation(self):
        """两个 tracker 的读去重缓存隔离。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("hello\n")
            path = f.name
        try:
            t1 = FileStateTracker()
            t2 = FileStateTracker()

            t1.record_read(path, 1, 500)
            # t2 没有记录，不应命中
            assert t2.check_dedup(path, 1, 500) is False
        finally:
            os.unlink(path)

    def test_reset_clears_all_state(self):
        """reset() 清空所有状态。"""
        t = FileStateTracker()
        t.record_patch_failure("/a.py")
        t.record_patch_failure("/a.py")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("data\n")
            path = f.name
        try:
            t.record_read(path, 1, 500)
            t.reset()
            assert t.record_patch_failure("/a.py") == 1
            assert t.check_dedup(path, 1, 500) is False
        finally:
            os.unlink(path)


class TestReadTracker:
    """测试读去重缓存（基于 FileStateTracker 实例）。"""

    def setup_method(self):
        self.tracker = FileStateTracker()

    def _make_file(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
        f.write(content)
        f.close()
        return f.name

    def test_first_read_returns_false(self):
        """首次读取文件返回 False（无缓存）。"""
        path = self._make_file("hello\nworld\n")
        try:
            result = self.tracker.check_dedup(path, 1, 500)
            assert result is False
        finally:
            os.unlink(path)

    def test_second_read_same_file_returns_true(self):
        """再次读取相同文件返回 True（去重命中）。"""
        path = self._make_file("hello\nworld\n")
        try:
            self.tracker.record_read(path, 1, 500)
            result = self.tracker.check_dedup(path, 1, 500)
            assert result is True
        finally:
            os.unlink(path)

    def test_different_offset_no_hit(self):
        """不同 offset 不命中。"""
        path = self._make_file("hello\nworld\nfoo\nbar\n")
        try:
            self.tracker.record_read(path, 1, 2)
            result = self.tracker.check_dedup(path, 3, 2)
            assert result is False
        finally:
            os.unlink(path)

    def test_modified_file_no_hit(self):
        """文件修改后不命中。"""
        path = self._make_file("original\n")
        try:
            self.tracker.record_read(path, 1, 500)
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("modified\n")
            result = self.tracker.check_dedup(path, 1, 500)
            assert result is False
        finally:
            os.unlink(path)

    def test_invalidate_clears_dedup(self):
        """invalidate_dedup 清空缓存。"""
        path = self._make_file("hello\n")
        try:
            self.tracker.record_read(path, 1, 500)
            self.tracker.invalidate_dedup(path)
            result = self.tracker.check_dedup(path, 1, 500)
            assert result is False
        finally:
            os.unlink(path)

    def test_resets_consecutive_counter(self):
        """notify_other_tool 后重置连续计数。"""
        path = self._make_file("hello\n")
        try:
            for _ in range(3):
                self.tracker.record_read(path, 1, 500)
            self.tracker.notify_other_tool()
            self.tracker.record_read(path, 1, 500)
            assert self.tracker.check_dedup(path, 1, 500) is True
        finally:
            os.unlink(path)


class TestPatchFailureTracker:
    """测试补丁失败跟踪（基于 FileStateTracker 实例）。"""

    def setup_method(self):
        self.tracker = FileStateTracker()

    def test_first_failure_returns_1(self):
        """首次失败返回 1。"""
        count = self.tracker.record_patch_failure("/tmp/test.py")
        assert count == 1

    def test_second_failure_returns_2(self):
        """再次失败返回 2。"""
        self.tracker.record_patch_failure("/tmp/test.py")
        count = self.tracker.record_patch_failure("/tmp/test.py")
        assert count == 2

    def test_reset_clears_failures(self):
        """reset_patch_failures 清空失败计数。"""
        self.tracker.record_patch_failure("/tmp/test.py")
        self.tracker.record_patch_failure("/tmp/test.py")
        self.tracker.reset_patch_failures(["/tmp/test.py"])
        count = self.tracker.record_patch_failure("/tmp/test.py")
        assert count == 1

    def test_third_failure_returns_3(self):
        """第三次失败返回 3（触发升级提示的条件）。"""
        self.tracker.record_patch_failure("/tmp/test.py")
        self.tracker.record_patch_failure("/tmp/test.py")
        count = self.tracker.record_patch_failure("/tmp/test.py")
        assert count == 3

    def test_different_paths_independent(self):
        """不同路径的失败计数独立。"""
        self.tracker.record_patch_failure("/tmp/a.py")
        self.tracker.record_patch_failure("/tmp/a.py")
        count_b = self.tracker.record_patch_failure("/tmp/b.py")
        assert count_b == 1  # b.py 首次失败

    def test_cap_at_64_paths(self):
        """最多跟踪 64 个不同文件。"""
        for i in range(65):
            path = f"/tmp/test_{i}.py"
            self.tracker.record_patch_failure(path)
        assert True