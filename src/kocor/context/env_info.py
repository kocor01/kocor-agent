"""动态环境信息收集。

在运行时自动收集当前环境信息，用于注入系统提示的 L3 层。
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import date


def build_environment_info() -> str:
    """构建动态环境信息块。

    收集：
    - 当前日期
    - 当前工作目录
    - Git 分支及工作区状态（轻量）
    - 操作系统信息

    设计决策：
    - 保持轻量（不超过 200 token）
    - 每次 Agent.run() / stream() 时重新收集
    - 不包含敏感信息（API key、密码等）

    Returns:
        格式化的环境信息字符串，每行一个信息项
    """
    parts = []

    # 当前日期
    parts.append(f"当前日期: {date.today().isoformat()}")

    # 当前工作目录
    cwd = os.getcwd()
    parts.append(f"当前工作目录: {cwd}")

    # Git 状态（轻量）
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if branch:
            parts.append(f"Git 分支: {branch}")

        has_changes = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if has_changes:
            parts.append("工作区有未提交的更改（git diff 可查看详情）")
    except Exception:
        pass

    # 操作系统
    parts.append(f"操作系统: {platform.system()} {platform.release()}")

    return "\n".join(parts)
