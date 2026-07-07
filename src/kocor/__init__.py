"""Kocor Agent - 小而美的 LLM 自主 Agent 助手"""

from __future__ import annotations

import io
import os
import sys

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("kocor-agent")
except Exception:
    __version__ = "0.0.1"

# ── 项目全局 UTF-8 模式 ──────────────────────────────────────────────────
# 所有文件 I/O 和子进程输出统一使用 UTF-8，避免 Windows GBK 编解码错误。
# 通过 patch io.text_encoding 让 open()、Path.read_text() 等默认使用 UTF-8。
# 子进程继承环境变量，当前进程 stdout/stderr 也强制使用 UTF-8。

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_original_text_encoding = io.text_encoding


def _utf8_text_encoding(encoding: str | None = None) -> str:
    """将 None 编码解析为 UTF-8 而非 locale 编码。"""
    if encoding is None:
        return "utf-8"
    return _original_text_encoding(encoding)


io.text_encoding = _utf8_text_encoding

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
