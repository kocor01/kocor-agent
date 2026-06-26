"""测试从 JSON 配置文件加载 Skill。"""

import json
import os
import tempfile

import pytest

from kocor.skill.types import InvokeStrategy, SkillType
from kocor.skill.skill_manager import SkillManager


class TestLoadFromConfig:
    """测试 load_from_config()"""

    def test_missing_file_is_noop(self):
        registry = SkillManager()
        registry.load_from_config("nonexistent_file.json")
        assert registry.list_skills(enabled_only=False) == []

    def test_empty_config(self):
        registry = SkillManager()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump({}, f)
            path = f.name
        try:
            registry.load_from_config(path)
            assert registry.list_skills(enabled_only=False) == []
        finally:
            os.unlink(path)

    def test_empty_skills(self):
        registry = SkillManager()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump({"skills": {}}, f)
            path = f.name
        try:
            registry.load_from_config(path)
            assert registry.list_skills(enabled_only=False) == []
        finally:
            os.unlink(path)

    def test_load_prompt_skill(self):
        registry = SkillManager()
        config = {
            "skills": {
                "review": {
                    "type": "prompt",
                    "invoke": "both",
                    "description": "Review code",
                    "prompt_template": "Review this: {user_input}",
                    "prompt_role": "user",
                    "category": "development",
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            registry.load_from_config(path)
            skill = registry.get("review")
            assert skill is not None
            assert skill.name == "review"
            assert skill.description == "Review code"
            assert skill.skill_type == SkillType.PROMPT
            assert skill.invoke_strategy == InvokeStrategy.BOTH
            assert skill.prompt_template == "Review this: {user_input}"
            assert skill.prompt_role == "user"
            assert skill.category == "development"
            assert skill.enabled is True
            assert skill.handler is None
        finally:
            os.unlink(path)

    def test_load_code_skill(self):
        registry = SkillManager()
        config = {
            "skills": {
                "greet": {
                    "type": "code",
                    "invoke": "slash",
                    "description": "Greet someone",
                    "module": "tests.skill.test_skill_config_loading",
                    "function": "_greet_handler",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "category": "utility",
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            registry.load_from_config(path)
            skill = registry.get("greet")
            assert skill is not None
            assert skill.name == "greet"
            assert skill.skill_type == SkillType.CODE
            assert skill.invoke_strategy == InvokeStrategy.SLASH
            assert skill.handler is not None
            assert callable(skill.handler)
            assert skill.handler("world") == "Hello, world!"
            assert skill.category == "utility"
        finally:
            os.unlink(path)

    def test_load_llm_only_skill(self):
        registry = SkillManager()
        config = {
            "skills": {
                "format": {
                    "type": "prompt",
                    "invoke": "llm",
                    "description": "Format code",
                    "prompt_template": "Format: {user_input}",
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            registry.load_from_config(path)
            skill = registry.get("format")
            assert skill is not None
            assert skill.invoke_strategy == InvokeStrategy.LLM
        finally:
            os.unlink(path)

    def test_load_disabled_skill(self):
        registry = SkillManager()
        config = {
            "skills": {
                "old": {
                    "type": "prompt",
                    "description": "Old skill",
                    "prompt_template": "x",
                    "enabled": False,
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            registry.load_from_config(path)
            skill = registry.get("old")
            assert skill is not None
            assert skill.enabled is False
        finally:
            os.unlink(path)

    def test_load_code_skill_missing_module(self):
        registry = SkillManager()
        config = {
            "skills": {
                "bad": {
                    "type": "code",
                    "description": "Bad skill",
                    "module": "nonexistent_module_xyz",
                    "function": "handler",
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            with pytest.raises(ImportError):
                registry.load_from_config(path)
        finally:
            os.unlink(path)

    def test_load_multiple_skills(self):
        registry = SkillManager()
        config = {
            "skills": {
                "a": {
                    "type": "prompt",
                    "description": "Skill A",
                    "prompt_template": "A: {user_input}",
                },
                "b": {
                    "type": "prompt",
                    "description": "Skill B",
                    "prompt_template": "B: {user_input}",
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            registry.load_from_config(path)
            assert len(registry.list_skills(enabled_only=False)) == 2
            assert registry.get("a") is not None
            assert registry.get("b") is not None
        finally:
            os.unlink(path)


def _greet_handler(user_input: str) -> str:
    """Handler for code skill test (referenced in JSON config)."""
    return f"Hello, {user_input}!"
