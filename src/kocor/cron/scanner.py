"""Cron Prompt 注入扫描器。

采用两层扫描策略：
1. 严格模式（用户 prompt 创建/更新时）
2. 宽松模式（技能组装后的完整 prompt）

参考 Hermes 的设计：https://hermes-agent.dev/cron-security
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 威胁模式定义
# ---------------------------------------------------------------------------

# 严格模式 — 应用于用户 prompt
_CRON_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
    (r'rm\s+-rf\s+/', "destructive_root_rm"),
]

# 宽松模式 — 应用于技能组装后的 prompt
_CRON_SKILL_ASSEMBLED_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
]

# 隐式 Unicode 字符
_CRON_INVISIBLE_CHARS = {
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

# Emoji 相邻码点范围
_EMOJI_NEIGHBOUR_CP_RANGES = (
    (0x1F000, 0x1FFFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x1F1E6, 0x1F1FF),
    (0x20E3, 0x20E3),
)


def _is_emoji_cp(cp: int) -> bool:
    """判断 Unicode 码点是否在 emoji 邻近范围内。"""
    return any(lo <= cp <= hi for lo, hi in _EMOJI_NEIGHBOUR_CP_RANGES)


def _zwj_has_emoji_neighbour(text: str, idx: int) -> bool:
    """判断 ZWJ 是否在 emoji 序列内部。"""
    left = idx - 1
    while left >= 0 and ord(text[left]) == 0xFE0F:
        left -= 1
    right = idx + 1
    while right < len(text) and ord(text[right]) == 0xFE0F:
        right += 1
    return (
        left >= 0 and right < len(text)
        and _is_emoji_cp(ord(text[left]))
        and _is_emoji_cp(ord(text[right]))
    )


def _strip_invisible_unicode(text: str) -> tuple[str, list[str]]:
    """剥离隐式 Unicode 字符，保留 emoji 中的合法 ZWJ。

    返回 (cleaned_text, removed_codepoints)。
    """
    if not text:
        return text, []
    removed: set[str] = set()
    cleaned: list[str] = []
    for idx, ch in enumerate(text):
        if ch in _CRON_INVISIBLE_CHARS:
            if ch == '‍' and _zwj_has_emoji_neighbour(text, idx):
                cleaned.append(ch)
                continue
            removed.add(f"U+{ord(ch):04X}")
            continue
        cleaned.append(ch)
    return ''.join(cleaned), sorted(removed)


# =============================================================================
# 公开 API
# =============================================================================


def scan_cron_prompt(prompt: str) -> str:
    """扫描用户提供的 cron prompt（严格模式）。

    Args:
        prompt: 用户 prompt

    Returns:
        空字符串表示通过，非空字符串为错误信息
    """
    if not prompt:
        return ""

    # 检查隐式 Unicode
    cleaned, removed = _strip_invisible_unicode(prompt)
    if removed:
        return (
            f"Blocked: prompt contains invisible unicode characters "
            f"({' '.join(removed)}) (possible injection)."
        )

    for pattern, pid in _CRON_THREAT_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return (
                f"Blocked: prompt matches threat pattern '{pid}'. "
                f"Cron prompts must not contain injection or exfiltration payloads."
            )

    return ""


def scan_cron_skill_assembled(assembled: str) -> tuple[str, str]:
    """扫描技能组装后的完整 prompt（宽松模式）。

    Args:
        assembled: 技能内容组装后的完整 prompt

    Returns:
        (cleaned_prompt, error) 二元组，error 为空表示通过
    """
    if not assembled:
        return "", ""

    # 隐式 Unicode 自动清洗（宽松模式不清除，改为清洗）
    cleaned, removed = _strip_invisible_unicode(assembled)
    if removed:
        logger.warning(
            "Cron 技能组装 prompt: 从技能内容中清除了 %d 个不可见 Unicode 字符 (%s)",
            len(removed), ", ".join(removed),
        )

    for pattern, pid in _CRON_SKILL_ASSEMBLED_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return cleaned, (
                f"Blocked: prompt matches threat pattern '{pid}'. "
                f"Cron prompts must not contain injection directives."
            )

    return cleaned, ""