"""文件状态管理模块。

提供跨工具调用的文件状态跟踪：
1. 读去重缓存：防止重复读取同一文件浪费 token
2. 连续读循环检测：防止 LLM 陷入重复读取死循环
3. 补丁失败跟踪：连续失败 ≥3 次时提示升级到 write_file

使用方式：
    tracker = FileStateTracker()
    tracker.record_read(path, offset, limit)
    tracker.check_dedup(path, offset, limit)
"""

from __future__ import annotations

import os
import threading

# ── 常量 ──────────────────────────────────────────────────────

_DEDUP_CAP = 500
_PATCH_FAILURES_CAP = 64


class FileStateTracker:
    """文件状态追踪器。

    追踪读去重状态和补丁失败计数。
    每个 ToolManager（即每个 Agent）持有一个实例，天然隔离。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dedup: dict[tuple, float] = {}
        self._dedup_hits: dict[tuple, int] = {}
        self._last_key: tuple | None = None
        self._consecutive: int = 0
        self._patch_failures: dict[str, int] = {}

    # ── 读去重 ────────────────────────────────────────────

    def check_dedup(self, resolved_path: str, offset: int, limit: int) -> bool:
        """检查文件是否可去重（未变更且已缓存）。

        Args:
            resolved_path: 已解析的绝对路径
            offset: 读取起始行
            limit: 读取行数

        Returns:
            如果文件未变且已缓存返回 True
        """
        key = (resolved_path, offset, limit)
        with self._lock:
            cached_mtime = self._dedup.get(key)
        if cached_mtime is None:
            return False
        try:
            current_mtime = os.path.getmtime(resolved_path)
        except OSError:
            return False
        if current_mtime == cached_mtime:
            with self._lock:
                self._dedup_hits[key] = self._dedup_hits.get(key, 0) + 1
            return True
        return False

    def record_read(self, resolved_path: str, offset: int, limit: int) -> dict:
        """记录一次读取操作，返回 {"consecutive": int}。

        Args:
            resolved_path: 已解析的绝对路径
            offset: 读取起始行
            limit: 读取行数

        Returns:
            包含了连续计数的字典
        """
        dedup_key = (resolved_path, offset, limit)
        read_key = ("read", resolved_path, offset, limit)

        with self._lock:
            # 更新 mtime 缓存
            try:
                mtime = os.path.getmtime(resolved_path)
                self._dedup[dedup_key] = mtime
                if len(self._dedup) > _DEDUP_CAP:
                    self._dedup.pop(next(iter(self._dedup)))
            except OSError:
                pass

            # 连续读取计数
            if self._last_key == read_key:
                self._consecutive += 1
            else:
                self._last_key = read_key
                self._consecutive = 1

            consecutive = self._consecutive

            # 如果这是真实读取（不是去重命中），重置 hits 计数
            self._dedup_hits.pop(dedup_key, None)

            return {"consecutive": consecutive}

    def get_dedup_hits(self, resolved_path: str, offset: int, limit: int) -> int:
        """获取去重命中次数。"""
        key = (resolved_path, offset, limit)
        with self._lock:
            return self._dedup_hits.get(key, 0)

    def get_consecutive_count(self) -> int:
        """获取连续读取计数。"""
        with self._lock:
            return self._consecutive

    def invalidate_dedup(self, resolved_path: str) -> None:
        """写入/补丁后清空指定路径的去重缓存。"""
        with self._lock:
            stale = [k for k in self._dedup if k[0] == resolved_path]
            for k in stale:
                del self._dedup[k]
            hit_keys = [k for k in self._dedup_hits if k[0] == resolved_path]
            for k in hit_keys:
                del self._dedup_hits[k]

    def notify_other_tool(self) -> None:
        """非读/搜工具调用时重置连续读取计数器。"""
        with self._lock:
            self._last_key = None
            self._consecutive = 0
            self._dedup_hits.clear()

    # ── 补丁失败跟踪 ─────────────────────────────────────

    def record_patch_failure(self, resolved_path: str) -> int:
        """记录一次补丁失败，返回该路径的连续失败次数。"""
        with self._lock:
            count = self._patch_failures.get(resolved_path, 0) + 1
            self._patch_failures[resolved_path] = count
            if len(self._patch_failures) > _PATCH_FAILURES_CAP:
                self._patch_failures.pop(next(iter(self._patch_failures)))
            return count

    def reset_patch_failures(self, resolved_paths: list[str]) -> None:
        """补丁成功后清空失败计数。"""
        if not resolved_paths:
            return
        with self._lock:
            for rp in resolved_paths:
                self._patch_failures.pop(rp, None)

    # ── 全量重置（会话切换时） ─────────────────────────────

    def reset(self) -> None:
        """清空所有追踪状态。"""
        with self._lock:
            self._dedup.clear()
            self._dedup_hits.clear()
            self._last_key = None
            self._consecutive = 0
            self._patch_failures.clear()


