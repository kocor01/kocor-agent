"""路径安全与环境变量清理。"""

from __future__ import annotations

import os


def resolve_safe_path(path: str, allowed_dir: str) -> str:
    """解析并校验路径是否在允许目录内，防止路径遍历攻击。

    如果 path 是绝对路径，直接使用（跳过 allowed_dir 锚定）；
    如果是相对路径，基于 allowed_dir 解析并验证。

    Args:
        path: 用户传入的路径
        allowed_dir: 允许的根目录

    Returns:
        归一化后的安全绝对路径

    Raises:
        PermissionError: 路径尝试逃逸到允许目录外
    """
    if os.path.isabs(path):
        resolved = os.path.realpath(path)
    else:
        resolved = os.path.realpath(os.path.join(allowed_dir, path))
        if resolved != allowed_dir and not resolved.startswith(allowed_dir + os.sep):
            raise PermissionError(f"Path traversal denied: {path}")
    return resolved


def sanitize_env() -> dict[str, str]:
    """创建不含敏感凭证的环境变量副本，防止子进程泄露 API Key。"""
    env = os.environ.copy()
    _sensitive_keys = [
        key
        for key in env
        if key.endswith(("_API_KEY", "_SECRET", "_TOKEN")) or key in ("OPENAI_ORG_ID",)
    ]
    for key in _sensitive_keys:
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env
