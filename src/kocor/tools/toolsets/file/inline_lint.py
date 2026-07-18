"""内联 Lint 模块。

提供零外部依赖的语法检查，支持：
- Python (ast.parse)
- JSON (json.loads)
- YAML (基本语法校验)
- TOML (tomllib/toml 解析)

用于 write_file 和 patch_file 后增量检查新引入的错误。
"""

from __future__ import annotations

import ast
import json
import os
import traceback
from typing import Any

LINTERS: dict[str, str] = {
    ".py": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def lint_content(filepath: str, content: str) -> dict[str, Any]:
    """对文件内容执行内联语法检查。

    Returns:
        {"status": "ok" | "error" | "skipped", "output": str}
    """
    ext = os.path.splitext(filepath)[1].lower()
    lint_type = LINTERS.get(ext)
    if lint_type is None:
        return {"status": "skipped", "output": ""}
    if not content.strip():
        return {"status": "ok", "output": ""}
    try:
        if lint_type == "python":
            ast.parse(content)
        elif lint_type == "json":
            json.loads(content)
        elif lint_type == "toml":
            _parse_toml(content)
        elif lint_type == "yaml":
            _check_yaml(content)
        return {"status": "ok", "output": ""}
    except (SyntaxError, ValueError, Exception) as e:
        return {"status": "error", "output": _format_error(e)}


def _format_error(exc: Exception) -> str:
    """格式化异常为单行错误文本。"""
    tb = traceback.format_exception_only(type(exc), exc)
    return "".join(tb).strip()


def _parse_toml(content: str) -> None:
    """TOML 格式校验（优先用 tomllib，回退到简易行解析）。"""
    try:
        import tomllib
        tomllib.loads(content)
        return
    except ImportError:
        pass
    except Exception:
        raise
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            raise ValueError(f"TOML parse error at line {i}: expected key=value")


def _check_yaml(content: str) -> None:
    """YAML 基本校验：缩进一致性 + 父行冒号检查。"""
    lines = content.splitlines()
    prev_indent = 0
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip())
        if indent > prev_indent and lines[i - 2].rstrip().endswith(":"):
            pass  # 正常缩进
        elif indent > prev_indent:
            raise ValueError(
                f"YAML indent error at line {i}: "
                "increased indent without parent ':'"
            )
        prev_indent = indent


def check_lint_delta(
    filepath: str,
    content_before: str,
    content_after: str,
) -> dict[str, Any] | None:
    """检查编辑后是否引入了新的 lint 错误。

    Returns:
        None 如果无新错误，或错误信息的 dict。
    """
    before = lint_content(filepath, content_before)
    after = lint_content(filepath, content_after)
    if after["status"] == "ok":
        return None
    if before["status"] == "error" and before["output"] == after["output"]:
        return None
    return after