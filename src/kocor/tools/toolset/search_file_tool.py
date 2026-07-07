"""search_files 内部工具。

支持内容搜索（rg/grep/Python 原生）和文件搜索，结果压缩以减少 token 消耗。
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from typing import Any

from kocor.tools.permission import PermissionManager
from kocor.tools.tool_utils import resolve_safe_path

logger = logging.getLogger(__name__)

_SEARCH_LOCK = threading.Lock()

_SEARCH_TIMEOUT = 15
_MAX_RESULTS = 200
_DENSIFY_MIN_MATCHES = 5


def _normalize_search_pagination(
    limit: int | None,
    offset: int | None,
) -> tuple[int, int]:
    if limit is None or limit < 1:
        limit = _MAX_RESULTS
    if offset is None or offset < 0:
        offset = 0
    return limit, offset


# ── Content search backends ──────────────────────────────────


def _search_with_rg(
    pattern: str, path: str, file_glob: str,
    limit: int, offset: int, output_mode: str, context: int,
) -> dict[str, Any]:
    cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
    if output_mode == "count":
        cmd.append("--count")
    elif output_mode == "files_only":
        cmd.append("--files-with-matches")
    if context > 0:
        cmd.extend(["-C", str(context)])
    if file_glob:
        cmd.extend(["--glob", file_glob])
    cmd.extend(["--max-count", str(limit + offset)])
    cmd.append(pattern)
    cmd.append(path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT)
    except FileNotFoundError:
        return {"error": "ripgrep (rg) not found"}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "timed_out": True}
    return _parse_rg_output(result, output_mode, limit, offset)


def _parse_rg_output(result, output_mode: str, limit: int, offset: int) -> dict[str, Any]:
    stdout = result.stdout
    if output_mode == "files_only":
        files = [f for f in stdout.splitlines() if f.strip()]
        return {"files": files[offset:offset + limit], "total_count": len(files)}
    if output_mode == "count":
        counts = {}
        for line in stdout.splitlines():
            if ":" in line:
                p, c = line.rsplit(":", 1)
                try:
                    counts[p] = int(c)
                except ValueError:
                    pass
        return {"counts": counts, "total_count": sum(counts.values())}
    return _parse_content_matches(stdout, limit, offset)


def _search_with_grep(
    pattern: str, path: str, file_glob: str,
    limit: int, offset: int, output_mode: str, context: int,
) -> dict[str, Any]:
    cmd = ["grep", "-rn"]
    if output_mode == "count":
        cmd.append("-c")
    elif output_mode == "files_only":
        cmd.append("-l")
    if context > 0:
        cmd.extend(["-C", str(context)])
    if file_glob:
        cmd.extend(["--include", file_glob])
    cmd.append(pattern)
    cmd.append(path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT)
    except FileNotFoundError:
        return {"error": "grep not found"}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "timed_out": True}
    if result.returncode == 2 and not result.stdout:
        return {"error": result.stderr.strip() or "grep error"}
    return _parse_content_matches(result.stdout, limit, offset)


def _search_with_python(
    pattern: str, path: str, file_glob: str,
    limit: int, offset: int, output_mode: str, context: int,
) -> dict[str, Any]:
    """Python 原生内容搜索（rg/grep 后备）。

    跨平台兼容，不需要外部命令。
    """
    if not os.path.isdir(path):
        return {"error": f"Not a directory: {path}"}

    matches = []
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    for root, _dirs, files in os.walk(path):
        # 跳过隐藏目录
        if "/." in root or "\\." in root:
            continue
        for fname in files:
            if file_glob:
                if not _match_glob(fname, file_glob):
                    continue
            fpath = os.path.join(root, fname)
            if output_mode == "files_only":
                matches.append({"path": os.path.relpath(fpath, path), "line": 0, "content": ""})
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if compiled.search(line):
                            matches.append({
                                "path": os.path.relpath(fpath, path),
                                "line": i,
                                "content": line.rstrip(),
                            })
                            if output_mode == "count":
                                # count mode: just count, don't accumulate content
                                pass
            except (OSError, UnicodeDecodeError):
                continue

    if output_mode == "count":
        file_counts = {}
        for m in matches:
            file_counts[m["path"]] = file_counts.get(m["path"], 0) + 1
        return {"counts": file_counts, "total_count": len(matches)}

    total = len(matches)
    page = matches[offset:offset + limit]

    if len(page) >= _DENSIFY_MIN_MATCHES:
        dense = _densify_matches(page)
        return {
            "matches_format": "path-grouped: each file path on its own line, followed by indented '<line>: <content>' rows",
            "matches_text": dense,
            "total_count": total,
        }
    return {"matches": page, "total_count": total}


def _match_glob(fname: str, pattern: str) -> bool:
    """简单的通配符匹配。"""
    if pattern == "*":
        return True
    if pattern.startswith("*."):
        return fname.endswith(pattern[1:])
    if pattern.endswith("*"):
        return fname.startswith(pattern[:-1])
    import fnmatch
    return fnmatch.fnmatch(fname, pattern)


# ── File search backends ────────────────────────────────────


def _search_files_with_rg(pattern: str, path: str, limit: int, offset: int) -> dict[str, Any]:
    cmd = ["rg", "--files", "--sortr=modified"]
    if pattern:
        cmd.extend(["--glob", f"*{pattern}*"])
    cmd.append(path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT)
    except FileNotFoundError:
        return {"error": "ripgrep not found"}
    except subprocess.TimeoutExpired:
        return {"error": "File search timed out"}
    files = [f for f in result.stdout.splitlines() if f.strip()]
    return {"files": files[offset:offset + limit], "total_count": len(files)}


def _search_files_with_find(pattern: str, path: str, limit: int, offset: int) -> dict[str, Any]:
    cmd = ["find", path, "-type", "f"]
    if pattern:
        cmd.extend(["-name", f"*{pattern}*"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT)
    except FileNotFoundError:
        return {"error": "find not found"}
    except subprocess.TimeoutExpired:
        return {"error": "File search timed out"}
    files = [f for f in result.stdout.splitlines() if f.strip()]
    rel = []
    for fname in files:
        rel.append(os.path.relpath(fname, path) if fname.startswith(path) else fname)
    return {"files": rel[offset:offset + limit], "total_count": len(rel)}


def _search_files_with_python(pattern: str, path: str, limit: int, offset: int) -> dict[str, Any]:
    """Python 原生文件搜索（rg/find 后备）。"""
    if not os.path.isdir(path):
        return {"error": f"Not a directory: {path}"}
    matched = []
    for root, _dirs, files in os.walk(path):
        if "/." in root or "\\." in root:
            continue
        for fname in files:
            if pattern and pattern.lower() not in fname.lower():
                continue
            matched.append(os.path.relpath(os.path.join(root, fname), path))
    matched.sort()
    return {"files": matched[offset:offset + limit], "total_count": len(matched)}


# ── Output parsing ──────────────────────────────────────────


def _parse_content_matches(output: str, limit: int, offset: int) -> dict[str, Any]:
    if not output.strip():
        return {"matches": [], "total_count": 0}
    matches = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split(":", 2)
        if len(parts) >= 3:
            p, ls, c = parts[0], parts[1], parts[2]
            try:
                ln = int(ls)
            except ValueError:
                ln = 0
            matches.append({"path": p, "line": ln, "content": c})
        elif len(parts) == 2:
            matches.append({"path": parts[0], "line": 0, "content": parts[1]})
    total = len(matches)
    page = matches[offset:offset + limit]
    if len(page) >= _DENSIFY_MIN_MATCHES:
        dense = _densify_matches(page)
        return {
            "matches_format": "path-grouped: each file path on its own line, followed by indented '<line>: <content>' rows",
            "matches_text": dense,
            "total_count": total,
        }
    return {"matches": page, "total_count": total}


def _densify_matches(matches: list[dict]) -> str:
    lines: list[str] = []
    cur: str | None = None
    for m in matches:
        if m["path"] != cur:
            lines.append(m["path"])
            cur = m["path"]
        lines.append(f"  {m['line']}: {m['content'].rstrip()}")
    return "\n".join(lines)


# ── Tool class ──────────────────────────────────────────────


class SearchFiles:
    """搜索文件和文件内容工具。"""

    NAME = "search_files"
    DESCRIPTION = (
        "搜索文件和文件内容。优先使用 ripgrep，回退到 grep 再到 Python 原生搜索。"
        "用于查找特定模式在代码库中的位置。"
    )
    SAFETY_LEVEL = PermissionManager.SAFETY_SAFE
    PARAMETERS = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "搜索模式（正则表达式或字符串）",
            },
            "target": {
                "type": "string",
                "enum": ["content", "files"],
                "description": "搜索目标：content=文件内容, files=文件名",
                "default": "content",
            },
            "path": {
                "type": "string",
                "description": "搜索目录（默认当前工作目录）",
                "default": ".",
            },
            "file_glob": {
                "type": "string",
                "description": "文件通配符过滤，如 *.py",
                "default": "",
            },
            "limit": {
                "type": "integer", "description": "最大结果数", "default": 50,
            },
            "offset": {
                "type": "integer", "description": "结果偏移", "default": 0,
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_only", "count"],
                "description": "输出模式", "default": "content",
            },
            "context": {
                "type": "integer", "description": "上下文行数", "default": 0,
            },
        },
        "required": ["pattern"],
    }

    @staticmethod
    def handler(
        pattern: str,
        target: str = "content",
        path: str = ".",
        file_glob: str = "",
        limit: int | None = None,
        offset: int | None = None,
        output_mode: str = "content",
        context: int = 0,
    ) -> str:
        try:
            limit, offset = _normalize_search_pagination(limit, offset)
            allowed_dir = os.path.realpath(os.getcwd())
            try:
                safe_path = resolve_safe_path(path, allowed_dir)
            except PermissionError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

            result: dict[str, Any] = {}

            if target == "content":
                result = _search_with_rg(pattern, safe_path, file_glob, limit, offset, output_mode, context)
                if "error" in result:
                    result = _search_with_grep(pattern, safe_path, file_glob, limit, offset, output_mode, context)
                if "error" in result:
                    result = _search_with_python(pattern, safe_path, file_glob, limit, offset, output_mode, context)

            elif target == "files":
                result = _search_files_with_rg(pattern, safe_path, limit, offset)
                if "error" in result:
                    result = _search_files_with_find(pattern, safe_path, limit, offset)
                if "error" in result or not result.get("files"):
                    result = _search_files_with_python(pattern, safe_path, limit, offset)
            else:
                return json.dumps({"error": f"Unknown target: {target}"}, ensure_ascii=False)

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.exception("search_files error")
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)