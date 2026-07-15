"""路径安全与环境变量清理。"""

from __future__ import annotations

import os


def resolve_safe_path(path: str, allowed_dir: str) -> str:
    """解析并校验路径是否在允许目录内，防止路径遍历攻击。

    无论 path 是绝对路径还是相对路径，都基于 allowed_dir 解析，
    并验证解析后的绝对路径未逃逸出 allowed_dir。
    绝对路径若指向 allowed_dir 外（如 /etc/passwd、C:\\Windows）将被拒绝，
    此前绝对路径直接放行的行为构成 P0.1 越权漏洞。

    Args:
        path: 用户传入的路径
        allowed_dir: 允许的根目录

    Returns:
        归一化后的安全绝对路径

    Raises:
        PermissionError: 路径尝试逃逸到允许目录外
    """
    # 统一规范化 allowed_dir，确保与 resolved 的比较口径一致
    # （调用方传入的 allowed_dir 可能未经 realpath 规范化）
    base = os.path.realpath(allowed_dir)
    # 所有路径都基于 base 解析：绝对路径会覆盖 base（os.path.join 语义），
    # 相对路径则拼接在 base 之下
    resolved = os.path.realpath(os.path.join(base, path))
    if resolved != base and not resolved.startswith(base + os.sep):
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
