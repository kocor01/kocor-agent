"""确定性会话键生成。"""

from __future__ import annotations

import os


def build_session_key(profile: str | None = None) -> str:
    """生成 Kocor 的确定性会话键。

    格式: ``kocor:{namespace}:cli``

    - profile 为 None/""/"default" 时使用环境变量 ``KOCOR_SESSION_NAME`` 或 "default"
    - 显式传入 profile 时优先使用传入值

    Args:
        profile: 会话命名空间标识

    Returns:
        格式化的会话键字符串
    """
    if not profile or profile == "default":
        profile = os.environ.get("KOCOR_SESSION_NAME") or "default"
    return f"kocor:{profile}:cli"
