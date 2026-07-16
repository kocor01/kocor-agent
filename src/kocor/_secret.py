"""敏感字符串容器，防止 repr/str/日志意外泄露 API Key 等凭证。"""

from __future__ import annotations

import hmac


class SecretStr:
    """防止 repr/str 意外泄露的敏感字符串容器。

    行为：
    - repr() / str() 输出掩码，不暴露实际值
    - .reveal() 显式获取原始值（调用方需注意使用场景）
    - eq 支持与 str 和 SecretStr 的恒定时间比较
    - bool(SecretStr("")) == False
    """

    _MASK = "******"

    def __init__(self, value: str):
        self._value = value

    def reveal(self) -> str:
        """显式获取原始值。"""
        return self._value

    def __repr__(self) -> str:
        return f"SecretStr('{self._MASK}')"

    def __str__(self) -> str:
        return self._MASK

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return hmac.compare_digest(
                self._value.encode("utf-8"),
                other._value.encode("utf-8"),
            )
        if isinstance(other, str):
            return hmac.compare_digest(
                self._value.encode("utf-8"),
                other.encode("utf-8"),
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __len__(self) -> int:
        return len(self._value)