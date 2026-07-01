"""系统提示构建器。

组装多层系统提示（L1 身份 + L2 项目指令 + L3 环境 + L4 记忆引导）。
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


_MEMORY_GUIDANCE = """## 记忆指引

应该记忆的（优先级从高到低）：
1. 用户偏好和纠正
2. 环境事实（框架版本、项目约定、工具特性）
3. 稳定的工作流约定

禁止记忆的：
- 任务进度、会话产出、临时 TODO
- PR 编号、issue 编号、commit SHA
- 7 天内会过时的内容

格式要求：
- ✅ User prefers concise responses（纯文本声明）
- ❌ Always respond concisely（祈使句会被当指令执行）"""


class SystemPromptBuilder:
    """多层系统提示构建器。

    接收可选的 MemoryStore，
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

        # L4: 记忆指引 + 持久快照（如有）
        if self.memory:
            layers.append(_MEMORY_GUIDANCE)
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

    def _build_memories_block(self) -> str:
        """构建持久记忆文本块（来自冻结快照）。"""
        if not self.memory:
            return ""
        return self.memory.format_for_system_prompt()