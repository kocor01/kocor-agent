"""测试目录发现技能。"""

import os
import tempfile

from kocor.skill.models import InvokeStrategy, SkillType
from kocor.skill.skill_manager import SkillManager


class TestDiscoverSkills:
    """测试 discover_skills()"""

    def test_nonexistent_directory_is_noop(self):
        registry = SkillManager()
        registry.discover_skills("/nonexistent_skill_dir_xyz")
        assert registry.list_skills(enabled_only=False) == []

    def test_discover_prompt_skill_file(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_code = (
                'NAME = "review"\n'
                'DESCRIPTION = "Review code"\n'
                'SKILL_TYPE = "prompt"\n'
                'INVOKE_STRATEGY = "both"\n'
                'PROMPT_TEMPLATE = "Review this:\\n{user_input}"\n'
                'PROMPT_ROLE = "user"\n'
                'CATEGORY = "development"\n'
            )
            with open(os.path.join(tmpdir, "review.py"), "w", encoding="utf-8") as f:
                f.write(skill_code)

            registry.discover_skills(tmpdir)
            skill = registry.get("review")
            assert skill is not None
            assert skill.name == "review"
            assert skill.description == "Review code"
            assert skill.skill_type == SkillType.PROMPT
            assert skill.invoke_strategy == InvokeStrategy.BOTH
            assert skill.prompt_template == "Review this:\n{user_input}"
            assert skill.prompt_role == "user"
            assert skill.category == "development"
            assert skill.enabled is True

    def test_discover_code_skill_file(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_code = (
                'NAME = "greet"\n'
                'DESCRIPTION = "Greet someone"\n'
                'SKILL_TYPE = "code"\n'
                'INVOKE_STRATEGY = "slash"\n'
                'PARAMETERS = {"type": "object", "properties": {"name": {"type": "string"}}}\n'
                'CATEGORY = "utility"\n'
                '\n'
                'def handler(user_input: str) -> str:\n'
                '    return f"Hello, {user_input}!"\n'
            )
            with open(os.path.join(tmpdir, "greet.py"), "w", encoding="utf-8") as f:
                f.write(skill_code)

            registry.discover_skills(tmpdir)
            skill = registry.get("greet")
            assert skill is not None
            assert skill.name == "greet"
            assert skill.skill_type == SkillType.CODE
            assert skill.invoke_strategy == InvokeStrategy.SLASH
            assert skill.handler is not None
            assert callable(skill.handler)
            assert skill.handler("world") == "Hello, world!"
            assert skill.category == "utility"

    def test_skip_private_files(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_code = (
                'NAME = "hidden"\n'
                'DESCRIPTION = "Hidden skill"\n'
                'SKILL_TYPE = "prompt"\n'
                'PROMPT_TEMPLATE = "x"\n'
            )
            with open(os.path.join(tmpdir, "_private.py"), "w", encoding="utf-8") as f:
                f.write(skill_code)

            registry.discover_skills(tmpdir)
            assert registry.get("hidden") is None

    def test_skip_file_without_name(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "no_name.py"), "w", encoding="utf-8") as f:
                f.write('DESCRIPTION = "No name"\n')

            registry.discover_skills(tmpdir)
            assert registry.list_skills(enabled_only=False) == []

    def test_discover_multiple_files(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, content in [
                ("a.py", 'NAME = "a"\nDESCRIPTION = "A"\nSKILL_TYPE = "prompt"\nPROMPT_TEMPLATE = "a"\n'),
                ("b.py", 'NAME = "b"\nDESCRIPTION = "B"\nSKILL_TYPE = "code"\ndef handler(x): return x\n'),
            ]:
                with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as f:
                    f.write(content)

            registry.discover_skills(tmpdir)
            assert len(registry.list_skills(enabled_only=False)) == 2
            assert registry.get("a") is not None
            assert registry.get("b") is not None

    def test_discover_with_defaults(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_code = (
                'NAME = "minimal"\n'
                'DESCRIPTION = "Minimal skill"\n'
            )
            with open(os.path.join(tmpdir, "minimal.py"), "w", encoding="utf-8") as f:
                f.write(skill_code)

            registry.discover_skills(tmpdir)
            skill = registry.get("minimal")
            assert skill is not None
            # Defaults for CODE type (SKILL_TYPE defaults to "code")
            assert skill.skill_type == SkillType.CODE
            assert skill.invoke_strategy == InvokeStrategy.BOTH
            assert skill.category == "discovered"
            assert skill.enabled is True

    def test_config_overrides_discovered(self):
        """配置文件优先于目录发现：同名时配置文件的 wins"""
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_code = (
                'NAME = "greet"\n'
                'DESCRIPTION = "Discovered greet"\n'
                'SKILL_TYPE = "code"\n'
                'def handler(x): return f"discovered {x}"\n'
            )
            with open(os.path.join(tmpdir, "greet.py"), "w", encoding="utf-8") as f:
                f.write(skill_code)

            import json

            config = {
                "skills": {
                    "greet": {
                        "type": "code",
                        "description": "Config greet",
                        "module": "tests.skill.test_skill_config_loading",
                        "function": "_greet_handler",
                    },
                },
            }
            config_path = os.path.join(tmpdir, "skills.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)

            # 先加载配置，再发现目录
            registry.load_from_config(config_path)
            registry.discover_skills(tmpdir)

            skill = registry.get("greet")
            assert skill is not None
            # 配置文件的描述优先
            assert skill.description == "Config greet"


class TestDiscoverClineSkills:
    """测试 discover_cline_skills() — Cline 格式 (SKILL.md + _meta.json)"""

    def test_nonexistent_directory_is_noop(self):
        registry = SkillManager()
        registry.discover_cline_skills("/nonexistent_xyz")
        assert registry.list_skills(enabled_only=False) == []

    def test_discover_skill_from_skill_md(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "weather")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(
                    "---\n"
                    'name: weather\n'
                    'description: Get current weather\n'
                    '---\n'
                    "\n"
                    "# Weather\n"
                    "Use `curl` to get weather.\n"
                )
            with open(os.path.join(skill_dir, "_meta.json"), "w", encoding="utf-8") as f:
                f.write('{"slug": "weather", "version": "1.0.0"}\n')

            registry.discover_cline_skills(tmpdir)
            skill = registry.get("weather")
            assert skill is not None
            assert skill.name == "weather"
            assert skill.description == "Get current weather"
            assert skill.skill_type == SkillType.PROMPT
            assert skill.invoke_strategy == InvokeStrategy.BOTH
            assert skill.category == "cline"
            assert skill.enabled is True
            assert "# Weather" in skill.prompt_template
            assert "Use `curl` to get weather." in skill.prompt_template

    def test_skip_if_no_skilli_md(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "empty_skill")
            os.makedirs(skill_dir)
            # 没有 SKILL.md 文件
            registry.discover_cline_skills(tmpdir)
            assert registry.list_skills(enabled_only=False) == []

    def test_skip_if_no_frontmatter(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "bad_skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write("# Just a heading\nNo frontmatter here.\n")
            registry.discover_cline_skills(tmpdir)
            assert registry.list_skills(enabled_only=False) == []

    def test_parse_full_frontmatter(self):
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "full_skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(
                    "---\n"
                    'name: demo\n'
                    'description: A demo skill\n'
                    'homepage: https://example.com\n'
                    "metadata: {\"key\": \"value\"}\n"
                    "---\n"
                    "\n"
                    "Body content here.\n"
                )
            registry.discover_cline_skills(tmpdir)
            skill = registry.get("demo")
            assert skill is not None
            assert skill.name == "demo"
            assert skill.description == "A demo skill"
            assert skill.prompt_template == "Body content here."

    def test_config_overrides_cline_skill(self):
        """配置文件中的同名 skill 优先于 Cline 格式发现的"""
        registry = SkillManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            import json

            config = {
                "skills": {
                    "weather": {
                        "type": "prompt",
                        "description": "Config weather",
                        "prompt_template": "Config version",
                    },
                },
            }
            config_path = os.path.join(tmpdir, "skills.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)

            skill_dir = os.path.join(tmpdir, "weather")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(
                    "---\n"
                    'name: weather\n'
                    'description: Cline weather\n'
                    "---\n"
                    "\n"
                    "Cline version body\n"
                )

            registry.load_from_config(config_path)
            registry.discover_cline_skills(tmpdir)

            skill = registry.get("weather")
            assert skill is not None
            assert skill.description == "Config weather"
            assert skill.prompt_template == "Config version"
