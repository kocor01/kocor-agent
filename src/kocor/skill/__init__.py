"""Skill 模块。"""

from __future__ import annotations

from kocor.skill.models import (
    InvokeStrategy,
    SkillContext,
    SkillDefinition,
    SkillResult,
    SkillType,
)
from kocor.skill.registry import SkillRegistry, skill

__all__ = [
    "InvokeStrategy",
    "SkillContext",
    "SkillDefinition",
    "SkillRegistry",
    "SkillResult",
    "SkillType",
    "skill",
]