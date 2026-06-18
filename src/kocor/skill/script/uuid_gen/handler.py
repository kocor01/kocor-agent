from __future__ import annotations

import uuid


def handler(user_input: str = "") -> str:
    """生成 UUID v4。"""
    return str(uuid.uuid4())