"""文件安全守卫模块。

提供多层安全保护：
1. 敏感路径阻断：阻止写入系统关键路径
2. 写拒绝列表：阻止写入凭证/密钥/缓存文件
3. 内容守卫：阻止将工具内部显示文本（如行号前缀内容）写回文件
"""

from __future__ import annotations

import os
import re
from pathlib import Path


# ── 敏感系统路径（跨平台，检查 path.parts） ───────────────────────
# 使用 Path.parts 元组进行检查以兼容 Windows/Linux。
# 仅靠 resolve_safe_path 的 allowed_dir 越界检查不足以覆盖所有场景
# （如 allowed_dir 恰好包含系统根），此处作为第二层防御。
_SENSITIVE_SYSTEM_ROOTS: frozenset[str] = frozenset({
    "etc",
    "boot",
    "usr",
    "bin",
    "sbin",
    "lib",
    "lib64",
    "var",
    "opt",
})


# ── 写拒绝路径构建 ───────────────────────────────────────────────
def _build_write_denied_paths() -> set[str]:
    """构建精确匹配的写拒绝路径集合。"""
    home = os.path.expanduser("~")
    return {
        os.path.realpath(p) if os.path.isabs(p) else os.path.realpath(os.path.join(home, p))
        for p in [
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            os.path.join(home, ".ssh", "id_ecdsa"),
            os.path.join(home, ".ssh", "id_dsa"),
            os.path.join(home, ".ssh", "config"),
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".git-credentials"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".pgpass"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".pypirc"),
        ]
    }


def _build_write_denied_prefixes() -> list[str]:
    """构建写拒绝目录前缀列表。"""
    home = os.path.expanduser("~")
    return [
        os.path.realpath(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            os.path.join(home, ".docker"),
            os.path.join(home, ".config", "gh"),
            os.path.join(home, ".config", "gcloud"),
            os.path.join(home, ".config", "autostart"),
        ]
    ]


# 缓存拒绝路径，避免每次调用都重新构建
_WRITE_DENIED_PATHS: set[str] = _build_write_denied_paths()
_WRITE_DENIED_PREFIXES: list[str] = _build_write_denied_prefixes()

# 项目级 .env 文件模式
_ENV_FILE_PATTERN = re.compile(r"\.env(\..+)?$")
# Shell 启动文件模式
_SHELL_RC_PATTERN = re.compile(r"\.(?:bashrc|bash_profile|bash_login|profile|zshrc|zprofile|zlogin|cshrc|tcshrc|kshrc|shrc)$")


def is_write_denied(path: str) -> bool:
    """检查路径是否在写拒绝列表中。

    Args:
        path: 要检查的绝对路径

    Returns:
        如果路径被拒绝写入返回 True
    """
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        return False

    # 精确匹配检查
    if resolved in _WRITE_DENIED_PATHS:
        return True

    # 前缀匹配检查
    for prefix in _WRITE_DENIED_PREFIXES:
        if resolved.startswith(prefix):
            return True

    # .env 文件检查（任意路径）
    filename = os.path.basename(resolved)
    if _ENV_FILE_PATTERN.search(filename):
        return True

    # Shell 启动文件检查（任意路径，防写入 ~/.bashrc 等）
    if _SHELL_RC_PATTERN.search(filename):
        return True

    return False


# ── 敏感路径检查 ─────────────────────────────────────────────────

def check_sensitive_path(
    path: str,
    allowed_dir: str | None = None,
) -> str | None:
    """检查路径是否指向敏感系统位置。

    Args:
        path: 文件路径（可以是通过 allowed_dir 解析的绝对路径）
        allowed_dir: 允许的根目录（用于解析相对路径）

    Returns:
        如果是敏感路径返回错误消息，否则返回 None
    """
    # 尝试解析为绝对路径
    if allowed_dir and not os.path.isabs(path):
        resolved = os.path.realpath(os.path.join(allowed_dir, path))
    elif os.path.isabs(path):
        resolved = os.path.realpath(path)
    else:
        resolved = os.path.realpath(os.path.expanduser(path))

    return _check_path_parts(resolved, path)


def _check_path_parts(resolved: str, original_path: str) -> str | None:
    """通过 Path.parts 检查路径是否为敏感系统路径。

    跨平台兼容：Windows 上 /etc/passwd 解析为 D:\\etc\\passwd，
    其 parts 为 ('D:\\', 'etc', 'passwd')。
    """
    try:
        parts = Path(resolved).parts
    except Exception:
        return None

    # 检查是否以敏感系统根目录开头（parts[1] 是第一个目录组件）
    # 例如 C:\etc\passwd → parts = ('C:\\', 'etc', 'passwd')
    if len(parts) >= 2 and parts[1] in _SENSITIVE_SYSTEM_ROOTS:
        return (
            f"Refusing to write to sensitive system path: {original_path}\n"
            "Use the terminal tool with sudo if you need to modify system files."
        )

    return None


# ── 可读路径阻断 ─────────────────────────────────────────────────

def get_read_block_error(path: str) -> str | None:
    """检查可读路径是否被阻断（凭证/缓存文件）。

    Args:
        path: 文件绝对路径

    Returns:
        如果是被阻断文件返回错误消息，否则返回 None
    """
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        return None

    # 拒绝读取 .env 文件
    filename = os.path.basename(resolved)
    if _ENV_FILE_PATTERN.search(filename):
        return "Reading .env files is blocked (may contain secrets)"

    # 拒绝读取 SSH 密钥等凭证文件
    if resolved in _WRITE_DENIED_PATHS:
        return "Reading credential files is blocked"

    for prefix in _WRITE_DENIED_PREFIXES:
        if resolved.startswith(prefix):
            return "Reading credential files is blocked"

    return None


# ── 内容守卫 ─────────────────────────────────────────────────────

_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current"
)


def _is_internal_file_status_text(content: str) -> bool:
    """检测内容是否为读去重的状态消息。"""
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped == _READ_DEDUP_STATUS_MESSAGE:
        return True
    if _READ_DEDUP_STATUS_MESSAGE in stripped and len(stripped) <= 2 * len(_READ_DEDUP_STATUS_MESSAGE):
        return True
    return False


def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """检测内容是否包含 read_file 的行号前缀格式。

    检测 `LINE_NUMBER|CONTENT` 格式的多行内容，
    防止 LLM 将带行号的显示文本写回文件。
    """
    if not isinstance(content, str):
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numbered: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        if "|" in stripped:
            prefix_part = stripped.split("|", 1)[0]
            if prefix_part.strip().isdigit():
                numbered.append(int(prefix_part.strip()))

    if len(numbered) < 2:
        return False
    if len(numbered) / len(lines) < 0.6:
        return False

    consecutive_pairs = sum(
        1 for prev, current in zip(numbered, numbered[1:])
        if current == prev + 1
    )
    return consecutive_pairs >= len(numbered) - 1


def is_internal_file_tool_content(content: str) -> bool:
    """检查内容是否为文件工具的内部显示文本而非真实文件内容。

    防止 LLM 将去重状态消息或行号前缀显示内容写回文件。
    """
    return (
        _is_internal_file_status_text(content)
        or _looks_like_read_file_line_numbered_content(content)
    )