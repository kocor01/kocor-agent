"""测试 SkillRegistry 核心功能。"""

import pytest

from kocor.skill.skill_manager import SkillManager
from kocor.skill.types import SkillDefinition, SkillType


class TestSkillRegistryRegister:
    """测试 register / get / list_skills"""

    def test_register_and_get(self):
        registry = SkillManager()
        skill = SkillDefinition(
            name="review",
            description="Review code",
            skill_type=SkillType.PROMPT,
            prompt_template="Review: {user_input}",
        )
        registry.register(skill)
        assert registry.get("review") is skill

    def test_register_duplicate_raises(self):
        registry = SkillManager()
        skill = SkillDefinition(
            name="review",
            description="Review code",
            skill_type=SkillType.PROMPT,
        )
        registry.register(skill)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(skill)

    def test_get_unknown_returns_none(self):
        registry = SkillManager()
        assert registry.get("nonexistent") is None

    def test_list_skills_all(self):
        registry = SkillManager()
        s1 = SkillDefinition(name="a", description="A", skill_type=SkillType.PROMPT)
        s2 = SkillDefinition(name="b", description="B", skill_type=SkillType.CODE)
        registry.register(s1)
        registry.register(s2)
        skills = registry.list_skills(enabled_only=False)
        assert len(skills) == 2

    def test_list_skills_enabled_only(self):
        registry = SkillManager()
        s1 = SkillDefinition(name="a", description="A", skill_type=SkillType.PROMPT, enabled=True)
        s2 = SkillDefinition(name="b", description="B", skill_type=SkillType.CODE, enabled=False)
        registry.register(s1)
        registry.register(s2)
        skills = registry.list_skills(enabled_only=True)
        assert len(skills) == 1
        assert skills[0].name == "a"

    def test_list_skills_by_category(self):
        registry = SkillManager()
        s1 = SkillDefinition(
            name="a", description="A", skill_type=SkillType.PROMPT, category="dev",
        )
        s2 = SkillDefinition(
            name="b", description="B", skill_type=SkillType.CODE, category="ops",
        )
        registry.register(s1)
        registry.register(s2)
        dev_skills = registry.list_skills(category="dev", enabled_only=False)
        assert len(dev_skills) == 1
        assert dev_skills[0].name == "a"

    def test_list_with_category_and_enabled(self):
        registry = SkillManager()
        s1 = SkillDefinition(
            name="a", description="A", skill_type=SkillType.PROMPT, category="dev", enabled=True,
        )
        s2 = SkillDefinition(
            name="b", description="B", skill_type=SkillType.PROMPT, category="dev", enabled=False,
        )
        registry.register(s1)
        registry.register(s2)
        skills = registry.list_skills(category="dev", enabled_only=True)
        assert len(skills) == 1
        assert skills[0].name == "a"

    def test_empty_registry_lists(self):
        registry = SkillManager()
        assert registry.list_skills(enabled_only=False) == []
        assert registry.get("anything") is None


class TestSkillRegistryInit:
    """测试 SkillRegistry 初始化"""

    def test_default_skills_empty(self):
        registry = SkillManager()
        assert registry.list_skills(enabled_only=False) == []

    def test_accepts_tool_manager(self):
        from kocor.tools.tool_manager import ToolManager
        tr = ToolManager()
        registry = SkillManager(tool_manager=tr)
        assert registry._tool_manager is tr