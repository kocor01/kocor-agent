"""write_file 内部工具。

支持原子写入、敏感路径守卫、行尾/BOM 保留、内容守卫。
"""

from __future__ import annotations

import json
import os
import tempfile

from kocor.tools.permission import PermissionManager
from kocor.tools.tool_utils import resolve_safe_path
from kocor.tools.toolsets.file.file_safety import (
    check_sensitive_path,
    is_internal_file_tool_content,
    is_write_denied,
)
from kocor.tools.toolsets.file.file_state import FileStateTracker

# ── 行尾/BOM 检测 ────────────────────────────────────────────────
_UTF8_BOM = "﻿"


def _detect_line_ending(sample: str) -> str | None:
    """检测文件的行尾风格。

    Args:
        sample: 文件内容采样（前 4096 字节）

    Returns:
        "\\r\\n" (CRLF) 或 "\\n" (LF) 或 None（无法确定）
    """
    if not sample:
        return None
    if "\r\n" in sample:
        return "\r\n"
    if "\n" in sample:
        return "\n"
    return None


def _has_bom(text: str) -> bool:
    """检查文本是否以 UTF-8 BOM 开头。"""
    return bool(text and text.startswith(_UTF8_BOM))


def _strip_bom(text: str) -> tuple[str, bool]:
    """剥离 UTF-8 BOM。"""
    if text and text.startswith(_UTF8_BOM):
        return text[1:], True
    return text, False


def _normalize_line_endings(text: str, target: str) -> str:
    """统一行尾风格。"""
    lf_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return lf_normalized
    if target == "\r\n":
        return lf_normalized.replace("\n", "\r\n")
    return text


# ── 工具类 ───────────────────────────────────────────────────────


class WriteFileTool:
    """写入文件内容工具。

    特性：
    - 原子写入（tempfile + os.replace）
    - 敏感路径检查
    - 写拒绝列表（凭证/密钥文件）
    - 内容守卫（阻止内部显示文本写回）
    - 行尾保留（CRLF/LF）
    - UTF-8 BOM 保留
    - 自动创建父目录
    """

    @classmethod
    def handler_factory(cls, **deps):
        """返回带 file_state 注入的 handler。"""
        fs_val = deps.get("file_state")
        return lambda **kw: WriteFileTool.handler(file_state=fs_val, **kw)

    NAME = "write_file"
    DESCRIPTION = "写入文件内容。自动创建父目录。支持原子写入防止文件损坏。"
    SAFETY_LEVEL = PermissionManager.SAFETY_DANGEROUS
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（绝对路径或相对于当前工作目录的相对路径）",
            },
            "content": {
                "type": "string",
                "description": "文件内容",
            },
        },
        "required": ["path", "content"],
    }

    @staticmethod
    def handler(
        path: str,
        content: str,
        file_state: FileStateTracker | None = None,
    ) -> str:
        """写入文件。

        Args:
            path: 文件路径
            content: 文件内容
            file_state: FileStateTracker 实例（由 ToolManager 注入）

        Returns:
            JSON 字符串，包含 bytes_written, dirs_created 或 error
        """
        if file_state is None:
            return json.dumps({"error": "Internal error: file_state is required"}, ensure_ascii=False)

        # ── 安全守卫 ────────────────────────────────────────────
        err = _check_write_safety(path, content)
        if err:
            return json.dumps({"error": err}, ensure_ascii=False)

        # ── 路径解析 ────────────────────────────────────────────
        allowed_dir = os.path.realpath(os.getcwd())
        try:
            safe_path = resolve_safe_path(path, allowed_dir)
        except PermissionError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        # ── 写拒绝列表检查 ──────────────────────────────────────
        if is_write_denied(safe_path):
            return json.dumps({
                "error": f"Refusing to write to blocked path: {path}",
            }, ensure_ascii=False)

        # ── 行尾/BOM 检测 + 读取前置内容（为 lint delta） ────────
        original_line_ending = None
        had_bom = False
        content_before = ""
        if os.path.exists(safe_path):
            try:
                with open(safe_path, "rb") as f:
                    raw_sample = f.read(4096)
                text_sample = raw_sample.decode("utf-8", errors="replace")
                original_line_ending = _detect_line_ending(text_sample)
                had_bom = _has_bom(text_sample)
                # 读取完整前置内容用于 lint delta
                with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                    content_before = f.read()
            except (OSError, UnicodeDecodeError):
                pass

        # ── 原子写入 ────────────────────────────────────────────
        try:
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            dirs_created = True
        except OSError as e:
            return json.dumps({
                "error": f"Cannot create directory: {e}",
            }, ensure_ascii=False)

        try:
            # 处理内容：行尾统一 + BOM 添加
            write_content = content
            if original_line_ending:
                write_content = _normalize_line_endings(write_content, original_line_ending)
            if had_bom:
                write_content = _UTF8_BOM + write_content

            # 使用临时文件 + os.replace 实现原子写入
            write_bytes = write_content.encode("utf-8")
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(safe_path) or ".",
                prefix=".kocor_write_",
                delete=False,
            ) as tmp:
                tmp.write(write_bytes)
                tmp_path = tmp.name

            os.replace(tmp_path, safe_path)

        except OSError as e:
            # 清理临时文件
            if "tmp_path" in locals():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return json.dumps({
                "error": f"Write failed: {e}",
            }, ensure_ascii=False)

        # ── 刷新读去重缓存 ──────────────────────────────────────
        file_state.invalidate_dedup(safe_path)

        # ── Lint delta 检查（仅报告新增错误） ────────────────────
        result_dict: dict = {
            "bytes_written": len(write_bytes),
            "dirs_created": dirs_created,
        }
        try:
            from kocor.tools.toolsets.file.inline_lint import check_lint_delta
            lint_err = check_lint_delta(safe_path, content_before, write_content)
            if lint_err:
                result_dict["lint"] = lint_err
        except Exception:
            pass  # Lint 失败不阻塞写入

        return json.dumps(result_dict, ensure_ascii=False)


def _check_write_safety(path: str, content: str) -> str | None:
    """执行写入前的安全检查。

    Args:
        path: 文件路径
        content: 文件内容

    Returns:
        错误消息或 None
    """
    # 敏感路径检查
    err = check_sensitive_path(path)
    if err:
        return err

    # 内容守卫
    if is_internal_file_tool_content(content):
        return (
            "Refusing to write internal file tool display text as file content. "
            "Strip read_file line-number prefixes or reconstruct the intended "
            "file contents before writing."
        )

    return None