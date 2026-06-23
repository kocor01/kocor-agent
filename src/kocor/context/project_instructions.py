"""项目指令加载。

从项目根目录的 KOCOR.md 文件加载项目级指令（类似 Claude Code 的 CLAUDE.md），
注入系统提示的 L2 层。
"""

from __future__ import annotations

import os
from pathlib import Path


def load_project_instructions(path: str = "KOCOR.md") -> str:
    """从文件加载项目指令。

    如果文件不存在或为空，返回空字符串。
    如果文件存在，返回格式化的指令文本块（含 ## 项目指令 标题）。

    Args:
        path: 项目指令文件路径，默认为 "KOCOR.md"

    Returns:
        格式化的项目指令文本，文件不存在或为空时返回 ""
    """
    if not path or not os.path.exists(path):
        return ""

    content = Path(path).read_text().strip()
    if not content:
        return ""

    return f"## 项目指令\n\n{content}"
