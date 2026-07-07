"""测试文件状态管理模块。"""

import os
import tempfile
import time

from kocor.tools.toolset.file_state import (
    check_dedup,
    invalidate_dedup,
    notify_other_tool_call,
    record_patch_failure,
    record_read,
    reset_patch_failures,
    reset_task_state,
)

TASK_ID = "test_task"


class TestReadTracker:
    """测试读去重缓存。"""

    def setup_method(self):
        reset_task_state(TASK_ID)

    def test_first_read_returns_false(self):
        """首次读取文件返回 False（无缓存）。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("hello\nworld\n")
            path = f.name
        try:
            result = check_dedup(TASK_ID, path, 1, 500)
            assert result is False
        finally:
            os.unlink(path)

    def test_second_read_same_file_returns_true(self):
        """再次读取相同文件返回 True（去重命中）。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("hello\nworld\n")
            path = f.name
        try:
            record_read(TASK_ID, path, 1, 500)
            result = check_dedup(TASK_ID, path, 1, 500)
            assert result is True
        finally:
            os.unlink(path)

    def test_different_offset_no_hit(self):
        """不同 offset 不命中。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("hello\nworld\nfoo\nbar\n")
            path = f.name
        try:
            record_read(TASK_ID, path, 1, 2)
            result = check_dedup(TASK_ID, path, 3, 2)
            assert result is False
        finally:
            os.unlink(path)

    def test_modified_file_no_hit(self):
        """文件修改后不命中。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("original\n")
            path = f.name
        try:
            record_read(TASK_ID, path, 1, 500)
            # 修改文件
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("modified\n")
            result = check_dedup(TASK_ID, path, 1, 500)
            assert result is False
        finally:
            os.unlink(path)

    def test_invalidate_clears_dedup(self):
        """invalidate_dedup 清空缓存。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("hello\n")
            path = f.name
        try:
            record_read(TASK_ID, path, 1, 500)
            invalidate_dedup(TASK_ID, path)
            result = check_dedup(TASK_ID, path, 1, 500)
            assert result is False
        finally:
            os.unlink(path)


class TestNotifyOtherToolCall:
    """测试 notify_other_tool_call 重置循环计数器。"""

    def setup_method(self):
        reset_task_state(TASK_ID)
        # 清理临时文件状态
        invalidate_dedup(TASK_ID, "irrelevant/init")

    def test_resets_consecutive_counter(self):
        """非读/搜工具调用后重置连续计数。"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("hello\n")
            path = f.name
        try:
            # 模拟 3 次连续读取
            for _ in range(3):
                record_read(TASK_ID, path, 1, 500)
            # 插入其他工具调用
            notify_other_tool_call(TASK_ID)
            # 再次读取应视为全新
            record_read(TASK_ID, path, 1, 500)
            # 此时 check_dedup 应命中（文件未变）
            assert check_dedup(TASK_ID, path, 1, 500) is True
        finally:
            os.unlink(path)


class TestPatchFailureTracker:
    """测试补丁失败跟踪。"""

    def setup_method(self):
        reset_task_state(TASK_ID)  # noqa: F821

    def test_first_failure_returns_1(self):
        """首次失败返回 1。"""
        count = record_patch_failure(TASK_ID, "/tmp/test.py")
        assert count == 1

    def test_second_failure_returns_2(self):
        """再次失败返回 2。"""
        record_patch_failure(TASK_ID, "/tmp/test.py")
        count = record_patch_failure(TASK_ID, "/tmp/test.py")
        assert count == 2

    def test_reset_clears_failures(self):
        """reset_patch_failures 清空失败计数。"""
        record_patch_failure(TASK_ID, "/tmp/test.py")
        record_patch_failure(TASK_ID, "/tmp/test.py")
        reset_patch_failures(TASK_ID, ["/tmp/test.py"])
        count = record_patch_failure(TASK_ID, "/tmp/test.py")
        assert count == 1

    def test_third_failure_returns_3(self):
        """第三次失败返回 3（触发升级提示的条件）。"""
        record_patch_failure(TASK_ID, "/tmp/test.py")
        record_patch_failure(TASK_ID, "/tmp/test.py")
        count = record_patch_failure(TASK_ID, "/tmp/test.py")
        assert count == 3

    def test_different_paths_independent(self):
        """不同路径的失败计数独立。"""
        record_patch_failure(TASK_ID, "/tmp/a.py")
        record_patch_failure(TASK_ID, "/tmp/a.py")
        count_b = record_patch_failure(TASK_ID, "/tmp/b.py")
        assert count_b == 1  # b.py 首次失败

    def test_cap_at_64_paths(self):
        """每任务最多跟踪 64 个不同文件。"""
        for i in range(65):
            path = f"/tmp/test_{i}.py"
            record_patch_failure(TASK_ID, path)
        # 验证未崩溃
        assert True