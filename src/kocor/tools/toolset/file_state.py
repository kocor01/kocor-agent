"""文件状态管理模块。

提供跨工具调用的文件状态跟踪：
1. 读去重缓存：防止重复读取同一文件浪费 token
2. 连续读循环检测：防止 LLM 陷入重复读取死循环
3. 补丁失败跟踪：连续失败 ≥3 次时提示升级到 write_file
"""

from __future__ import annotations

import os
import threading
from typing import Any

# ── 读去重缓存 ─────────────────────────────────────────────────
#
# _read_tracker[task_id] = {
#     "dedup": {(resolved_path, offset, limit): mtime},      # 去重缓存
#     "dedup_hits": {(path, offset, limit): count},           # 连续命中次数
#     "last_key": (path, offset, limit) | None,              # 上次读取
#     "consecutive": int,                                     # 连续相同 key 计数
# }
#
# 容量限制：每任务最多 500 条去重缓存条目
# 超出时按插入顺序淘汰最旧条目（Python 3.7+ dict 有序）

_read_tracker_lock = threading.Lock()
_read_tracker: dict[str, dict[str, Any]] = {}

_DEDUP_CAP = 500
_MAX_CONSECUTIVE_READS = 4
_MAX_DEDUP_HITS = 2


def check_dedup(task_id: str, resolved_path: str, offset: int, limit: int) -> bool:
    """检查文件是否可去重（未变更且已缓存）。

    Args:
        task_id: 任务标识
        resolved_path: 已解析的绝对路径
        offset: 读取起始行
        limit: 读取行数

    Returns:
        如果文件未变且已缓存返回 True
    """
    dedup_key = (resolved_path, offset, limit)
    with _read_tracker_lock:
        task_data = _read_tracker.setdefault(task_id, {
            "dedup": {},
            "dedup_hits": {},
            "last_key": None,
            "consecutive": 0,
        })
        cached_mtime = task_data["dedup"].get(dedup_key)

    if cached_mtime is None:
        return False

    try:
        current_mtime = os.path.getmtime(resolved_path)
    except OSError:
        return False

    if current_mtime == cached_mtime:
        # 去重命中，记录命中次数
        with _read_tracker_lock:
            hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
            task_data["dedup_hits"][dedup_key] = hits
        return True

    return False


def record_read(task_id: str, resolved_path: str, offset: int, limit: int) -> dict:
    """记录一次读取操作。

    更新去重缓存和连续读取计数器。

    Args:
        task_id: 任务标识
        resolved_path: 已解析的绝对路径
        offset: 读取起始行
        limit: 读取行数

    Returns:
        包含了连续计数和命中计数的字典：
        {"consecutive": int, "dedup_hits": int}
    """
    dedup_key = (resolved_path, offset, limit)
    read_key = ("read", resolved_path, offset, limit)

    with _read_tracker_lock:
        task_data = _read_tracker.setdefault(task_id, {
            "dedup": {},
            "dedup_hits": {},
            "last_key": None,
            "consecutive": 0,
        })

        # 更新 mtime 缓存
        try:
            mtime = os.path.getmtime(resolved_path)
            task_data["dedup"][dedup_key] = mtime
            # 容量限制
            if len(task_data["dedup"]) > _DEDUP_CAP:
                task_data["dedup"].pop(next(iter(task_data["dedup"])))
        except OSError:
            pass

        # 连续读取计数
        if task_data["last_key"] == read_key:
            task_data["consecutive"] += 1
        else:
            task_data["last_key"] = read_key
            task_data["consecutive"] = 1

        consecutive = task_data["consecutive"]

        # 如果这是真实读取（不是去重命中），重置 hits 计数
        task_data["dedup_hits"].pop(dedup_key, None)

        return {
            "consecutive": consecutive,
        }


def get_dedup_hits(task_id: str, resolved_path: str, offset: int, limit: int) -> int:
    """获取去重命中次数。

    Args:
        task_id: 任务标识
        resolved_path: 已解析的绝对路径
        offset: 读取起始行
        limit: 读取行数

    Returns:
        去重命中次数
    """
    dedup_key = (resolved_path, offset, limit)
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return 0
        return task_data["dedup_hits"].get(dedup_key, 0)


def get_consecutive_count(task_id: str) -> int:
    """获取连续读取计数。"""
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return 0
        return task_data.get("consecutive", 0)


def invalidate_dedup(task_id: str, resolved_path: str) -> None:
    """写入/补丁后清空指定路径的去重缓存。

    使后续 read_file 总是返回新内容而非 "unchanged" 标记。

    Args:
        task_id: 任务标识
        resolved_path: 已解析的绝对路径
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return
        dedup = task_data.get("dedup")
        if not dedup:
            return
        stale_keys = [k for k in dedup if k[0] == resolved_path]
        for k in stale_keys:
            del dedup[k]
        # 同时清空 hits
        dedup_hits = task_data.get("dedup_hits", {})
        hit_keys = [k for k in dedup_hits if k[0] == resolved_path]
        for k in hit_keys:
            del dedup_hits[k]


def notify_other_tool_call(task_id: str) -> None:
    """非读/搜工具调用时重置连续读取计数器。

    确保只阻断真正连续的重复读取——如果 Agent 调用了其他工具
    （write/patch/bash 等），连续读计数重置。
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0
            task_data["dedup_hits"].clear()


# ── 补丁失败跟踪 ─────────────────────────────────────────────────
#
# _patch_failure_tracker[task_id] = {resolved_path: consecutive_failure_count}
# 每任务最多跟踪 64 个不同文件
# 成功补丁后通过 reset_patch_failures 清空

_patch_failure_lock = threading.Lock()
_patch_failure_tracker: dict[str, dict[str, int]] = {}
_PATCH_FAILURES_CAP = 64


def record_patch_failure(task_id: str, resolved_path: str) -> int:
    """记录一次补丁失败。

    Args:
        task_id: 任务标识
        resolved_path: 已解析的绝对路径

    Returns:
        该路径的连续失败次数
    """
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.setdefault(task_id, {})
        count = task_failures.get(resolved_path, 0) + 1
        task_failures[resolved_path] = count

        # 容量限制
        if len(task_failures) > _PATCH_FAILURES_CAP:
            task_failures.pop(next(iter(task_failures)))

        return count


def reset_patch_failures(task_id: str, resolved_paths: list[str]) -> None:
    """补丁成功后清空失败计数。

    Args:
        task_id: 任务标识
        resolved_paths: 已成功的路径列表
    """
    if not resolved_paths:
        return
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.get(task_id)
        if task_failures is None:
            return
        for rp in resolved_paths:
            task_failures.pop(rp, None)


def reset_task_state(task_id: str) -> None:
    """清空任务的文件状态（会话重置时调用）。"""
    with _read_tracker_lock:
        _read_tracker.pop(task_id, None)
    with _patch_failure_lock:
        _patch_failure_tracker.pop(task_id, None)