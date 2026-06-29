"""系统提示构建器。

组装多层系统提示（L1 身份 + L2 项目指令 + L3 环境 + L4 记忆）。
"""

from __future__ import annotations

import os
import platform
from datetime import date
from pathlib import Path
from typing import Any

from kocor.config import Config


def load_project_instructions(path: str = "KOCOR.md") -> str:
    """从文件加载项目指令。"""
    if not path or not os.path.exists(path):
        return ""

    content = Path(path).read_text().strip()
    if not content:
        return ""

    return f"## 项目指令\n\n{content}"


def build_environment_info() -> str:
    """构建动态环境信息块。"""
    parts = ["## 环境信息"]
    parts.append(f"当前日期: {date.today().isoformat()}")
    parts.append(f"当前工作目录: {os.getcwd()}")
    parts.append(f"操作系统: {platform.system()} {platform.release()}")
    return "\n".join(parts)


class SystemPromptBuilder:
    """多层系统提示构建器。

    接收可选的记忆管理器，
    组装含 L1-L4 所有层的完整系统提示文本。
    """

    def __init__(self, memory: Any = None):
        self.identity_prompt = Config.get("default_system_prompt")
        self.memory = memory
    def build(self) -> str:
        """构建完整的系统提示文本。"""
        layers = []

        # L1: 身份提示
        layers.append(self.identity_prompt)

        # L2: 项目指令
        project_instructions = self._load_project_instructions()
        if project_instructions:
            layers.append(project_instructions)

        # L3: 动态环境信息
        layers.append(self._build_environment_info())

        # L4: 持久记忆（如有）
        memories_text = self._build_memories_block()
        if memories_text:
            layers.append(memories_text)

        return "\n\n---\n\n".join(layers)


    @staticmethod
    def _load_project_instructions(path: str = "KOCOR.md") -> str:
        """从文件加载项目指令。"""
        if not path or not os.path.exists(path):
            return ""

        content = Path(path).read_text().strip()
        if not content:
            return ""

        return f"## 项目指令\n\n{content}"

    @staticmethod
    def _build_environment_info() -> str:
        """构建动态环境信息块。"""
        parts = ["## 环境信息"]
        parts.append(f"当前日期: {date.today().isoformat()}")
        parts.append(f"当前工作目录: {os.getcwd()}")
        parts.append(f"操作系统: {platform.system()} {platform.release()}")
        return "\n".join(parts)

    def _build_memories_block(self, max_items: int = 20) -> str:
        """构建持久记忆文本块。"""
        if not self.memory:
            return ""

        items = self.memory.list()[:max_items]
        if not items:
            return ""

        lines = ["## 已记录的信息\n"]
        for item in items:
            lines.append(f"### {item.name}")
            lines.append(item.content)
            lines.append("")

        return "\n".join(lines)