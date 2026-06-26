"""动态环境信息收集。

在运行时自动收集当前环境信息，用于注入系统提示的 L3 层。
"""

from __future__ import annotations

import os
import platform
from datetime import date


def build_environment_info() -> str:
    """构建动态环境信息块。

    收集：
    - 当前日期
    - 当前工作目录
    - 操作系统信息

    Returns:
        包含标题和内容的格式化字符串
    """
    parts = ["## 环境信息"]

    # 当前日期
    parts.append(f"当前日期: {date.today().isoformat()}")

    # 当前工作目录
    cwd = os.getcwd()
    parts.append(f"当前工作目录: {cwd}")

    # 操作系统
    parts.append(f"操作系统: {platform.system()} {platform.release()}")

    return "\n".join(parts)
