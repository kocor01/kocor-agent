"""威胁模式扫描。

提供 strict scope 的正则模式集合，用于记忆内容的安全检查。
- 写入时：每条新内容扫描通过才允许写入。
- 快照生成时：匹配条目替换为 [BLOCKED: ...] 占位符。

与 cron scanner (tools/toolsets/cron/scanner.py) 保持同步，
确保记忆写入路径与 cron prompt 路径有相同的注入防护。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ThreatMatch:
    """单次威胁匹配结果。"""

    pattern_name: str
    matched_text: str
    start: int
    end: int


# 隐式 Unicode 字符（与 cron scanner 同步）
_INVISIBLE_CHARS: set[str] = {
    '​',  # ZERO WIDTH SPACE
    '‌',  # ZERO WIDTH NON-JOINER
    '‍',  # ZERO WIDTH JOINER
    '‎',  # LEFT-TO-RIGHT MARK
    '‏',  # RIGHT-TO-LEFT MARK
    '⁠',  # WORD JOINER
    '⁡',  # FUNCTION APPLICATION
    '⁢',  # INVISIBLE TIMES
    '⁣',  # INVISIBLE SEPARATOR
    '⁤',  # INVISIBLE PLUS
    '⁦',  # LEFT-TO-RIGHT ISOLATE
    '⁧',  # RIGHT-TO-LEFT ISOLATE
    '⁨',  # FIRST STRONG ISOLATE
    '⁩',  # POP DIRECTIONAL ISOLATE
    '‪',  # LEFT-TO-RIGHT EMBEDDING
    '‫',  # RIGHT-TO-LEFT EMBEDDING
    '‬',  # POP DIRECTIONAL FORMATTING
    '‭',  # LEFT-TO-RIGHT OVERRIDE
    '‮',  # RIGHT-TO-LEFT OVERRIDE
}


# 严格模式模式集合（与 cron scanner 严格模式对齐）
_STRICT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "prompt_injection_ignore",
        re.compile(
            r"ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions?",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_injection_disregard",
        re.compile(
            r"disregard\s+(?:your|all|any|prior|previous)\s+(instructions|rules|guidelines)",
            re.IGNORECASE,
        ),
    ),
    (
        "deception_hide",
        re.compile(r"do\s+not\s+tell\s+the\s+user", re.IGNORECASE),
    ),
    (
        "sys_prompt_override",
        re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
    ),
    (
        "read_secrets",
        re.compile(r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)", re.IGNORECASE),
    ),
    (
        "script_tag",
        re.compile(r"<script\b[^>]*>", re.IGNORECASE),
    ),
    (
        "rm_rf_root",
        re.compile(r"rm\s+-rf\s+/(?:\s|$)", re.IGNORECASE),
    ),
]


def _has_invisible_chars(content: str) -> bool:
    """检查是否包含隐式 Unicode 字符（可能用于注入绕过）。"""
    for ch in content:
        if ch in _INVISIBLE_CHARS:
            return True
    return False


def scan_strict(content: str) -> list[ThreatMatch]:
    """扫描 strict scope 威胁模式。

    Args:
        content: 待扫描文本

    Returns:
        所有匹配的 ThreatMatch 列表（按位置排序），空列表表示安全。
    """
    if not content:
        return []

    matches: list[ThreatMatch] = []

    # 检查不可见 Unicode 字符
    if _has_invisible_chars(content):
        matches.append(
            ThreatMatch(
                pattern_name="invisible_unicode",
                matched_text="",
                start=0,
                end=0,
            )
        )

    for name, pattern in _STRICT_PATTERNS:
        for m in pattern.finditer(content):
            matches.append(
                ThreatMatch(
                    pattern_name=name,
                    matched_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                )
            )

    matches.sort(key=lambda x: (x.start, x.end))
    return matches


def is_safe_strict(content: str) -> bool:
    """快捷判断：内容是否通过 strict 扫描（无任何匹配）。"""
    return not scan_strict(content)
