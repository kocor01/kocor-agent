"""patch_file 内部工具。

使用模糊匹配替换文件中的文本块，生成 unified diff 输出。
"""

from __future__ import annotations

import difflib
import json
import logging
import os

from kocor.tools.permission import PermissionManager
from kocor.tools.tool_utils import resolve_safe_path
from kocor.tools.toolsets.file.file_safety import (
    check_sensitive_path,
    is_internal_file_tool_content,
    is_write_denied,
)
from kocor.tools.toolsets.file.file_state import FileStateTracker
from kocor.tools.toolsets.file.fuzzy_match import fuzzy_find_and_replace

logger = logging.getLogger(__name__)


class PatchFile:
    """替换文件中的文本块（补丁）工具。

    使用 6 策略模糊匹配链定位要替换的代码块，
    生成 unified diff 输出供 LLM 确认变更。

    特性：
    - 6 策略模糊匹配（精确 → 逐行去空白 → 合并空白 → 忽略缩进 → 边界松弛 → 块锚定）
    - 替换全部（replace_all）
    - 生成 unified diff
    - 写后验证
    - 补丁失败跟踪（连续 3 次失败提示升级到 write_file）
    - 安全守卫（敏感路径、写拒绝、内容守卫）
    """

    @classmethod
    def handler_factory(cls, **deps):
        """返回带 file_state 注入的 handler。"""
        fs_val = deps.get("file_state")
        return lambda **kw: PatchFile.handler(file_state=fs_val, **kw)

    NAME = "patch_file"
    DESCRIPTION = (
        "替换文件中精确匹配的文本块。"
        "支持模糊匹配（自动处理缩进/空白差异）。"
        "使用 replace_all=True 替换所有匹配。"
        "如果连续 3 次补丁失败，请改用 write_file 覆盖整个文件。"
    )
    SAFETY_LEVEL = PermissionManager.SAFETY_DANGEROUS
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "old_string": {
                "type": "string",
                "description": "要替换的旧文本（支持多行）",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的新文本（支持多行）",
            },
            "replace_all": {
                "type": "boolean",
                "description": "是否替换所有匹配（默认只替换第一个唯一匹配）",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    @staticmethod
    def handler(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        file_state: FileStateTracker | None = None,
    ) -> str:
        """执行模糊匹配替换。

        Args:
            path: 文件路径
            old_string: 要替换的旧文本
            new_string: 替换后的新文本
            replace_all: 是否替换所有匹配
            file_state: FileStateTracker 实例（由 ToolManager 注入）

        Returns:
            JSON 字符串
        """
        if file_state is None:
            return json.dumps({"error": "Internal error: file_state is required", "success": False}, ensure_ascii=False)

        try:
            # ── 安全守卫 ──────────────────────────────────────
            err = _check_patch_safety(path, old_string, new_string)
            if err:
                return json.dumps({"error": err, "success": False}, ensure_ascii=False)

            # ── 路径解析 ──────────────────────────────────────
            allowed_dir = os.path.realpath(os.getcwd())
            try:
                safe_path = resolve_safe_path(path, allowed_dir)
            except PermissionError as e:
                return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

            # ── 写拒绝列表检查 ────────────────────────────────
            if is_write_denied(safe_path):
                return json.dumps({
                    "error": f"Refusing to patch blocked path: {path}",
                    "success": False,
                }, ensure_ascii=False)

            # ── 读取文件 ──────────────────────────────────────
            if not os.path.exists(safe_path):
                return json.dumps({
                    "error": f"File not found: {path}",
                    "success": False,
                }, ensure_ascii=False)

            try:
                with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                return json.dumps({
                    "error": f"Cannot read file: {e}",
                    "success": False,
                }, ensure_ascii=False)

            # ── 模糊匹配替换 ──────────────────────────────────
            new_content, count, strategy, match_err = fuzzy_find_and_replace(
                content, old_string, new_string, replace_all,
            )

            if match_err or count == 0:
                err_msg = match_err or "No match found"
                # 记录补丁失败（用于升级提示）
                failures = file_state.record_patch_failure(safe_path)
                if failures >= 3:
                    err_msg += (
                        ". This patch has failed 3+ times consecutively. "
                        "Rewrite will not help — re-read the file with read_file "
                        "and use write_file to overwrite the entire file instead."
                    )
                return json.dumps({
                    "error": err_msg,
                    "success": False,
                }, ensure_ascii=False)

            # ── 生成 unified diff ──────────────────────────────
            diff_lines = list(difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            ))
            diff = "".join(diff_lines)

            # ── 写文件 ────────────────────────────────────────
            try:
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except OSError as e:
                return json.dumps({
                    "error": f"Write failed: {e}",
                    "success": False,
                }, ensure_ascii=False)

            # ── 写后验证 ──────────────────────────────────────
            try:
                with open(safe_path, "r", encoding="utf-8") as f:
                    verify_content = f.read()
                if verify_content != new_content:
                    return json.dumps({
                        "error": "Write verification failed: content mismatch",
                        "success": False,
                    }, ensure_ascii=False)
            except OSError as e:
                return json.dumps({
                    "error": f"Verify failed: {e}",
                    "success": False,
                }, ensure_ascii=False)

            # ── 刷新状态 ──────────────────────────────────────
            file_state.invalidate_dedup(safe_path)
            file_state.reset_patch_failures([safe_path])

            # ── Lint delta 检查（仅报告新增错误） ──────────────
            lint_result = None
            try:
                from kocor.tools.toolsets.file.inline_lint import check_lint_delta
                lint_result = check_lint_delta(safe_path, content, new_content)
            except Exception:
                pass

            result_dict = {
                "success": True,
                "diff": diff,
                "files_modified": [path],
                "strategy": strategy,
            }
            if lint_result:
                result_dict["lint"] = lint_result
            return json.dumps(result_dict, ensure_ascii=False)

        except Exception as e:
            logger.exception("patch_file error")
            return json.dumps({
                "error": f"{type(e).__name__}: {e}",
                "success": False,
            }, ensure_ascii=False)


def _check_patch_safety(path: str, old_string: str, new_string: str) -> str | None:
    """补丁前的安全检查。

    Returns:
        错误消息或 None
    """
    # 敏感路径检查
    err = check_sensitive_path(path)
    if err:
        return err

    # 内容守卫：检查 new_string 是否为工具内部显示文本
    if is_internal_file_tool_content(new_string):
        return (
            "Refusing to write internal file tool display text as file content. "
            "Strip read_file line-number prefixes from the replacement text."
        )

    return None