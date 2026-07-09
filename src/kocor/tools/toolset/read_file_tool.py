"""read_file 内部工具。

支持分页读取、行号前缀、去重检测、循环检测、二进制检测。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from kocor.tools.permission import PermissionManager
from kocor.tools.toolset.binary_extensions import has_binary_extension
from kocor.tools.toolset.read_extract import is_extractable_document
from kocor.tools.toolset.file_safety import get_read_block_error
from kocor.tools.toolset.file_state import FileStateTracker
from kocor.config import Config
from kocor.tools.tool_utils import resolve_safe_path

logger = logging.getLogger(__name__)

# ── 常量配置（通过 Config 读取） ────────────────────────────────
def _get_read_chars():
    try:
        return Config.load().file_read_max_chars
    except Exception:
        return 100_000

def _get_read_lines():
    try:
        return Config.load().file_read_max_lines
    except Exception:
        return 500
_LARGE_FILE_HINT_BYTES = 512_000  # 512 KB - 大文件提示阈值

# 设备路径黑名单
_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})

# 去重状态消息
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current"
)


def _is_blocked_device(path: str) -> bool:
    """检查是否为会挂起进程的设备路径。

    纯路径检查，不执行 I/O。
    """
    if not path:
        return False
    normalized = os.path.normpath(os.path.expanduser(path))
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/ 路径检查
    if normalized.startswith("/proc/") and normalized.endswith(
        ("/environ", "/cmdline", "/maps", "/smaps", "/fd/0", "/fd/1", "/fd/2")
    ):
        return True
    return False


def _normalize_pagination(offset: int | None, limit: int | None) -> tuple[int, int]:
    """归一化分页参数。

    Args:
        offset: 起始行号
        limit: 最大行数

    Returns:
        (normalized_offset, normalized_limit)
    """
    if offset is None or offset < 1:
        offset = 1
    if limit is None or limit < 1:
        limit = _get_read_lines()
    return offset, limit


# ── 工具类 ───────────────────────────────────────────────────────


class ReadFile:
    """读取文件内容工具。

    特性：
    - 分页读取（offset/limit）
    - 行号前缀
    - 去重缓存（相同文件+相同范围+未修改 → "unchanged"）
    - 连续读循环检测（4 次相同 → 硬阻断）
    - 二进制扩展名检测
    - 设备路径阻断
    - 凭证/缓存文件阻断
    - 字符数截断（防上下文窗口溢出）
    - 大文件提示
    """

    NAME = "read_file"
    DESCRIPTION = (
        "读取文件内容。支持分页读取，大文件请使用 offset 和 limit 参数。"
    )
    SAFETY_LEVEL = PermissionManager.SAFETY_SAFE
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（绝对路径或相对于当前工作目录的相对路径）",
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（从 1 开始，默认 1）",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "最大读取行数（默认 500）",
                "default": 500,
            },
        },
        "required": ["path"],
    }

    @staticmethod
    def handler(
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        file_state: FileStateTracker | None = None,
    ) -> str:
        """读取文件内容。

        Args:
            path: 文件路径
            offset: 起始行号（从 1 开始）
            limit: 最大行数
            file_state: FileStateTracker 实例（由 ToolManager 注入）

        Returns:
            JSON 字符串
        """
        if file_state is None:
            return json.dumps({"error": "Internal error: file_state is required"}, ensure_ascii=False)

        try:
            offset, limit = _normalize_pagination(offset, limit)

            # ── 设备路径检查 ──────────────────────────────────
            if _is_blocked_device(path):
                return json.dumps({
                    "error": (
                        f"Cannot read '{path}': this is a device file "
                        "that would block or produce infinite output."
                    ),
                }, ensure_ascii=False)

            # ── 路径解析 ──────────────────────────────────────
            allowed_dir = os.path.realpath(os.getcwd())
            try:
                safe_path = resolve_safe_path(path, allowed_dir)
            except PermissionError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

            # ── 文件存在性检查 ────────────────────────────────
            if not os.path.exists(safe_path):
                return json.dumps({
                    "error": f"File not found: {path}",
                }, ensure_ascii=False)

            # ── 二进制扩展名检测 ──────────────────────────────
            if has_binary_extension(safe_path):
                return json.dumps({
                    "error": (
                        f"Cannot read binary file '{path}' "
                        f"({Path(safe_path).suffix.lower()}). "
                        "Use vision tools for images, or bash for binary files."
                    ),
                }, ensure_ascii=False)

            # ── 读取阻断检查 ──────────────────────────────────
            block_error = get_read_block_error(safe_path)
            if block_error:
                return json.dumps({"error": block_error}, ensure_ascii=False)

            # ── 去重检查 ──────────────────────────────────────
            if file_state.check_dedup(safe_path, offset, limit):
                dedup_hits = file_state.get_dedup_hits(safe_path, offset, limit)
                if dedup_hits >= 2:
                    return json.dumps({
                        "error": (
                            f"BLOCKED: You have called read_file on this "
                            f"exact region {dedup_hits + 1} times and the file has "
                            "NOT changed. STOP calling read_file for this path "
                            "— the content from your earlier result is still current."
                        ),
                        "path": path,
                        "already_read": dedup_hits + 1,
                    }, ensure_ascii=False)
                return json.dumps({
                    "status": "unchanged",
                    "message": _READ_DEDUP_STATUS_MESSAGE,
                    "path": path,
                }, ensure_ascii=False)

            # ── 读取文件 ──────────────────────────────────────
            try:
                with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.read().splitlines(keepends=False)
            except (OSError, UnicodeDecodeError) as e:
                return json.dumps({
                    "error": f"Cannot read file: {e}",
                }, ensure_ascii=False)

            total_lines = len(all_lines)

            # 分页切片
            end_line = min(offset + limit - 1, total_lines)
            page_lines = all_lines[offset - 1 : end_line]

            # 添加行号前缀
            numbered_lines = []
            for i, line in enumerate(page_lines):
                line_num = offset + i
                numbered_lines.append(f"{line_num}|{line}")

            content = "\n".join(numbered_lines)
            content_len = len(content)
            try:
                file_size = os.path.getsize(safe_path)
            except OSError:
                file_size = 0

            truncated = end_line < total_lines
            result_dict = {
                "content": content,
                "total_lines": total_lines,
                "file_size": file_size,
                "truncated": truncated,
            }

            if truncated:
                result_dict["hint"] = (
                    f"File has {total_lines} lines, showing {offset}-{end_line}. "
                    f"Use offset={end_line + 1} to continue reading."
                )

            # ── 字符数守卫 ────────────────────────────────────
            if content_len > _get_read_chars():
                # 截断到最后一个完整行
                kept_lines = []
                running = 0
                for line in numbered_lines:
                    addition = len(line) + (1 if kept_lines else 0)
                    if running + addition > _get_read_chars():
                        break
                    kept_lines.append(line)
                    running += addition

                if not kept_lines:
                    kept_lines.append(numbered_lines[0][:_get_read_chars()])

                content = "\n".join(kept_lines)
                lines_kept = len(kept_lines)
                next_offset = offset + lines_kept
                shown_end = offset + lines_kept - 1
                result_dict["content"] = content
                result_dict["truncated"] = True
                result_dict["truncated_by"] = "chars"
                result_dict["next_offset"] = next_offset
                result_dict["hint"] = (
                    f"Output truncated at the {_get_read_chars():,}-char read budget "
                    f"after {lines_kept} line(s) (showing lines {offset}-"
                    f"{shown_end} of {total_lines}). Use offset={next_offset} "
                    "to continue."
                )

            # ── 大文件提示 ────────────────────────────────────
            if file_size > _LARGE_FILE_HINT_BYTES and limit > 200 and truncated:
                result_dict["hint"] = (
                    f"This file is large ({file_size:,} bytes). "
                    "Consider reading only the section you need with offset and limit."
                )

            # ── 记录读取状态 ──────────────────────────────────
            file_state.record_read(safe_path, offset, limit)

            # ── 连续循环检测 ──────────────────────────────────
            consecutive = file_state.get_consecutive_count()
            if consecutive >= 4:
                return json.dumps({
                    "error": (
                        f"BLOCKED: You have read this exact file region "
                        f"{consecutive} times in a row. "
                        "The content has NOT changed. "
                        "STOP re-reading and proceed with your task."
                    ),
                    "path": path,
                    "already_read": consecutive,
                }, ensure_ascii=False)
            elif consecutive >= 3:
                result_dict["_warning"] = (
                    f"You have read this exact file region {consecutive} times "
                    "consecutively. Use the information you already have."
                )

            return json.dumps(result_dict, ensure_ascii=False)

        except Exception as e:
            logger.exception("read_file error")
            return json.dumps({
                "error": f"{type(e).__name__}: {e}",
            }, ensure_ascii=False)


