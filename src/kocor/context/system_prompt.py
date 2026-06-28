"""系统提示构建器。

组装多层系统提示（L1 身份 + L2 项目指令 + L3 环境 + L4 记忆）。
"""

from __future__ import annotations

from typing import Any

from kocor.context.env_info import build_environment_info
from kocor.context.project_instructions import load_project_instructions


class SystemPromptBuilder:
    """多层系统提示构建器。

    接收 L1 身份提示和可选的记忆管理器，
    组装含 L1-L4 所有层的完整系统提示文本。
    """

    def __init__(self, identity_prompt: str, memory: Any = None):
        self.identity_prompt = identity_prompt
        self.memory = memory

    def build(self) -> str:
        """构建完整的系统提示文本。"""
        layers = []

        # L1: 身份提示
        layers.append(self.identity_prompt)

        # L2: 项目指令
        project_instructions = load_project_instructions()
        if project_instructions:
            layers.append(project_instructions)

        # L3: 动态环境信息
        layers.append(build_environment_info())

        # L4: 持久记忆（如有）
        memories_text = self._build_memories_block()
        if memories_text:
            layers.append(memories_text)

        return "\n\n---\n\n".join(layers)

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