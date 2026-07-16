"""Skill 模块。"""

from __future__ import annotations

from kocor.skill.skill_manager import SkillManager, skill
from kocor.skill.types import (
    InvokeStrategy,
    SkillContext,
    SkillDefinition,
    SkillResult,
    SkillType,
)

__all__ = [
    "InvokeStrategy",
    "SkillContext",
    "SkillDefinition",
    "SkillManager",
    "SkillResult",
    "SkillType",
    "skill",
]