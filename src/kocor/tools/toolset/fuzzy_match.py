"""模糊匹配模块。

实现 6 策略渐进式匹配链，用于 patch_file 工具的精确定位替换。
灵感来自 OpenCode 和 Hermes 的 9 策略匹配链。

策略链（按宽松度排序）：
1. exact — 精确字符串匹配
2. line_trimmed — 逐行去除首尾空白
3. whitespace_normalized — 合并连续空白
4. indentation_flexible — 忽略缩进
5. trimmed_boundary — 仅首尾行去空白
6. block_anchor — 首尾行锚定，中间行相似度
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable


# ── 策略函数 ───────────────────────────────────────────────────


def _strategy_exact(content: str, old_string: str, **_kwargs) -> bool:
    """策略 1：精确匹配。"""
    return old_string in content


def _strategy_line_trimmed(content: str, old_string: str, **_kwargs) -> bool:
    """策略 2：逐行去除首尾空白后比较。"""
    content_lines = [line.strip() for line in content.split("\n")]
    old_lines = [line.strip() for line in old_string.split("\n")]
    return _list_contains_sublist(content_lines, old_lines)


def _strategy_whitespace_normalized(content: str, old_string: str, **_kwargs) -> bool:
    """策略 3：合并连续空白（空格、制表符）后比较。"""
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()
    return _normalize(old_string) in _normalize(content)


def _strategy_indentation_flexible(content: str, old_string: str, **_kwargs) -> bool:
    """策略 4：剥离所有行首缩进后比较。"""
    def _strip_indent(s: str) -> str:
        return "\n".join(line.lstrip() for line in s.split("\n"))
    return _strip_indent(old_string) in _strip_indent(content)


def _strategy_trimmed_boundary(content: str, old_string: str, **_kwargs) -> bool:
    """策略 5：仅首尾行去除空白，中间行精确匹配。"""
    content_lines = content.split("\n")
    old_lines = old_string.split("\n")

    if len(old_lines) < 2:
        return False

    # 首尾行 strip，中间行精确匹配
    first_stripped = old_lines[0].strip()
    last_stripped = old_lines[-1].strip()

    for i in range(len(content_lines) - len(old_lines) + 1):
        if content_lines[i].strip() == first_stripped and \
           content_lines[i + len(old_lines) - 1].strip() == last_stripped:
            # 中间行完全匹配
            mid_match = True
            for j in range(1, len(old_lines) - 1):
                if content_lines[i + j] != old_lines[j]:
                    mid_match = False
                    break
            if mid_match:
                return True
    return False


def _strategy_block_anchor(content: str, old_string: str, **_kwargs) -> bool:
    """策略 6：首尾行锚定，中间行相似度 ≥80%。"""
    content_lines = content.split("\n")
    old_lines = old_string.split("\n")

    if len(old_lines) < 3:
        return False

    first_stripped = old_lines[0].strip()
    last_stripped = old_lines[-1].strip()

    for i in range(len(content_lines) - len(old_lines) + 1):
        if content_lines[i].strip() == first_stripped and \
           content_lines[i + len(old_lines) - 1].strip() == last_stripped:
            # 中间行逐行相似度检查
            match_count = 0
            total = len(old_lines) - 2
            for j in range(1, len(old_lines) - 1):
                cl = content_lines[i + j].strip()
                ol = old_lines[j].strip()
                if cl == ol or SequenceMatcher(None, cl, ol).ratio() >= 0.8:
                    match_count += 1
            if total > 0 and match_count / total >= 0.8:
                return True
    return False


# ── 策略注册表 ─────────────────────────────────────────────────

STRATEGIES: list[tuple[str, Callable]] = [
    ("exact", _strategy_exact),
    ("line_trimmed", _strategy_line_trimmed),
    ("whitespace_normalized", _strategy_whitespace_normalized),
    ("indentation_flexible", _strategy_indentation_flexible),
    ("trimmed_boundary", _strategy_trimmed_boundary),
    ("block_anchor", _strategy_block_anchor),
]

match_strategies = {name: func for name, func in STRATEGIES}


# ── 辅助函数 ───────────────────────────────────────────────────


def _list_contains_sublist(lst: list[str], sub: list[str]) -> bool:
    """检查列表 lst 是否包含子列表 sub。"""
    if len(sub) == 0:
        return True
    if len(sub) > len(lst):
        return False
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i + len(sub)] == sub:
            return True
    return False


def _reindent_replacement(
    content_region: str,
    old_string: str,
    new_string: str,
) -> str:
    """将 new_string 的缩进调整为匹配区域的缩进级别。

    当非精确匹配时，LLM 的缩进可能不同于文件实际缩进。
    此函数通过比较旧字符串匹配区域与实际内容的缩进差异来调整新字符串。
    """
    content_lines = content_region.splitlines()
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    if len(old_lines) != len(new_lines):
        return new_string

    adjusted = []
    for cl, ol, nl in zip(content_lines, old_lines, new_lines):
        old_indent = len(ol) - len(ol.lstrip())
        new_indent = len(nl) - len(nl.lstrip())
        content_indent = len(cl) - len(cl.lstrip())
        # 缩进差异：实际内容缩进 - 旧字符串缩进
        indent_diff = content_indent - old_indent
        adjusted_indent = max(0, new_indent + indent_diff)
        adjusted.append(" " * adjusted_indent + nl.lstrip())

    return "\n".join(adjusted)


# ── 主函数 ─────────────────────────────────────────────────────


def fuzzy_find_and_replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[str, int, str | None, str | None]:
    """使用渐进式模糊匹配链查找并替换文本。

    Args:
        content: 文件内容
        old_string: 要查找的文本
        new_string: 替换文本
        replace_all: 是否替换所有匹配

    Returns:
        (new_content, match_count, strategy_name, error_message)
        - 成功: (modified_content, 替换次数, 使用的策略, None)
        - 失败: (original_content, 0, None, 错误描述)
    """
    if not old_string:
        return content, 0, None, "old_string cannot be empty"

    if old_string == new_string:
        return content, 0, None, "old_string and new_string are identical"

    # 尝试每个策略
    for strategy_name, strategy_func in STRATEGIES:
        if not strategy_func(content, old_string=old_string):
            continue

        # 策略匹配成功，执行替换
        try:
            new_content, count = _do_replace(
                content, old_string, new_string, strategy_name, replace_all
            )
        except ValueError:
            continue

        if count > 0:
            return new_content, count, strategy_name, None

    return content, 0, None, "No match found with any strategy"


def _do_replace(
    content: str,
    old_string: str,
    new_string: str,
    strategy: str,
    replace_all: bool,
) -> tuple[str, int]:
    """执行实际替换操作。

    Args:
        content: 原始内容
        old_string: 要查找的文本
        new_string: 替换文本
        strategy: 使用的匹配策略
        replace_all: 是否替换所有

    Returns:
        (new_content, count)

    Raises:
        ValueError: 替换失败
    """
    if strategy == "exact":
        return _replace_exact(content, old_string, new_string, replace_all)
    else:
        return _replace_fuzzy(content, old_string, new_string, strategy, replace_all)


def _replace_exact(
    content: str, old_string: str, new_string: str, replace_all: bool
) -> tuple[str, int]:
    """精确替换。"""
    if replace_all:
        count = content.count(old_string)
        if count == 0:
            raise ValueError("No match found")
        return content.replace(old_string, new_string), count
    else:
        idx = content.find(old_string)
        if idx == -1:
            raise ValueError("No match found")
        # 检查唯一性
        second_idx = content.find(old_string, idx + 1)
        if second_idx != -1:
            raise ValueError("Multiple matches found, use replace_all=True")
        return content[:idx] + new_string + content[idx + len(old_string):], 1


def _replace_fuzzy(
    content: str,
    old_string: str,
    new_string: str,
    strategy: str,
    replace_all: bool,
) -> tuple[str, int]:
    """模糊替换。"""
    content_lines = content.splitlines()
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    # 找到匹配区域
    matches = _find_matches(content_lines, old_lines, strategy)

    if not matches:
        raise ValueError("No match found")

    if not replace_all and len(matches) > 1:
        raise ValueError("Multiple matches found, use replace_all=True")

    # 从后往前替换（保持索引正确）
    result_lines = list(content_lines)
    for start, end in reversed(matches):
        matched_region = "\n".join(content_lines[start:end])
        # 缩进对齐
        adjusted_new = _reindent_replacement(
            matched_region, old_string, new_string
        )
        result_lines[start:end] = adjusted_new.split("\n")

    return "\n".join(result_lines), len(matches)


def _find_matches(
    content_lines: list[str],
    old_lines: list[str],
    strategy: str,
) -> list[tuple[int, int]]:
    """在内容行序列中查找所有匹配区域。

    Returns:
        [(start_line, end_line), ...] 列表
    """
    # 剥离 \r 以确保跨平台行尾兼容
    content_lines = [line.rstrip("\r") for line in content_lines]
    old_lines = [line.rstrip("\r") for line in old_lines]
    matches = []
    i = 0
    while i <= len(content_lines) - len(old_lines):
        window = "\n".join(content_lines[i:i + len(old_lines)])

        # 用对应策略验证匹配
        strategy_func = match_strategies[strategy]
        if strategy_func(window, old_string="\n".join(old_lines)):
            matches.append((i, i + len(old_lines)))
            i += len(old_lines)  # 跳过已匹配区域
        else:
            i += 1

    return matches