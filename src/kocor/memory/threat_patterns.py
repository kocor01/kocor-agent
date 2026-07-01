"""威胁模式扫描。

提供 strict scope 的正则模式集合，用于记忆内容的安全检查。
- 写入时：每条新内容扫描通过才允许写入。
- 快照生成时：匹配条目替换为 [BLOCKED: ...] 占位符。
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


# strict scope 模式集合（v1 最小集）
_STRICT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "prompt_injection_ignore",
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions?", re.IGNORECASE),
    ),
    (
        "prompt_injection_disregard",
        re.compile(r"disregard\s+(?:all\s+)?(?:prior|previous)\s+instructions?", re.IGNORECASE),
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
