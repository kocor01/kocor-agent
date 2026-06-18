"""测试技能执行。"""

from kocor.skill.models import InvokeStrategy, SkillContext, SkillDefinition, SkillResult, SkillType
from kocor.skill.registry import SkillRegistry
from kocor.tool_registry import ToolRegistry


class TestExecutePromptSkill:
    """测试 PROMPT 类型技能执行"""

    def test_render_prompt_template(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="review",
            description="Review code",
            skill_type=SkillType.PROMPT,
            prompt_template="Review this code:\n{user_input}",
        )
        registry.register(skill)

        ctx = SkillContext(user_input="def foo(): pass")
        result = registry.execute("review", ctx)

        assert result.success is True
        assert result.content == "Review this code:\ndef foo(): pass"
        assert result.skill_name == "review"

    def test_prompt_with_system_role(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="expert",
            description="Expert mode",
            skill_type=SkillType.PROMPT,
            prompt_template="You are an expert in {user_input}",
            prompt_role="system",
        )
        registry.register(skill)

        ctx = SkillContext(user_input="Python")
        result = registry.execute("expert", ctx)

        assert result.success is True
        assert result.content == "You are an expert in Python"

    def test_prompt_with_extra_context(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="format",
            description="Format with context",
            skill_type=SkillType.PROMPT,
            prompt_template="{greeting}, {user_input}!",
        )
        registry.register(skill)

        ctx = SkillContext(user_input="world", extra={"greeting": "Hello"})
        result = registry.execute("format", ctx)

        assert result.content == "Hello, world!"

    def test_prompt_skill_name_in_template(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="test",
            description="Test",
            skill_type=SkillType.PROMPT,
            prompt_template="Skill: {skill_name}, Input: {user_input}",
        )
        registry.register(skill)

        ctx = SkillContext(user_input="hello")
        result = registry.execute("test", ctx)

        assert result.content == "Skill: test, Input: hello"


class TestExecuteCodeSkill:
    """测试 CODE 类型技能执行"""

    def test_code_skill_calls_handler(self):
        registry = SkillRegistry()

        def greet_handler(user_input: str) -> str:
            return f"Hello, {user_input}!"

        skill = SkillDefinition(
            name="greet",
            description="Greet",
            skill_type=SkillType.CODE,
            handler=greet_handler,
        )
        registry.register(skill)

        ctx = SkillContext(user_input="world")
        result = registry.execute("greet", ctx)

        assert result.success is True
        assert result.content == "Hello, world!"

    def test_code_skill_with_tools_param(self):
        registry = SkillRegistry()

        def tools_handler(user_input: str, tools: ToolRegistry) -> str:
            return f"tools: {len(tools.get_definitions())}"

        skill = SkillDefinition(
            name="check_tools",
            description="Check tools",
            skill_type=SkillType.CODE,
            handler=tools_handler,
        )
        registry.register(skill)

        tr = ToolRegistry()
        tr.register("dummy", "Dummy", {"type": "object"}, lambda **kw: "ok")
        ctx = SkillContext(user_input="x", tool_registry=tr)
        result = registry.execute("check_tools", ctx)

        assert result.content == "tools: 1"

    def test_code_skill_with_context_param(self):
        registry = SkillRegistry()

        def ctx_handler(context: SkillContext) -> str:
            return f"input: {context.user_input}"

        skill = SkillDefinition(
            name="ctx_test",
            description="Context test",
            skill_type=SkillType.CODE,
            handler=ctx_handler,
        )
        registry.register(skill)

        ctx = SkillContext(user_input="test")
        result = registry.execute("ctx_test", ctx)

        assert result.content == "input: test"

    def test_code_skill_no_handler(self):
        registry = SkillRegistry()

        skill = SkillDefinition(
            name="no_handler",
            description="No handler",
            skill_type=SkillType.CODE,
            handler=None,
        )
        registry.register(skill)

        ctx = SkillContext(user_input="x")
        result = registry.execute("no_handler", ctx)

        assert result.success is False
        assert "no handler" in result.content.lower()

    def test_code_skill_handler_exception(self):
        registry = SkillRegistry()

        def broken_handler(user_input: str) -> str:
            raise RuntimeError("something broke")

        skill = SkillDefinition(
            name="broken",
            description="Broken",
            skill_type=SkillType.CODE,
            handler=broken_handler,
        )
        registry.register(skill)

        ctx = SkillContext(user_input="x")
        result = registry.execute("broken", ctx)

        assert result.success is False
        assert "something broke" in result.content


class TestExecuteErrors:
    """测试执行错误情况"""

    def test_unknown_skill(self):
        registry = SkillRegistry()
        ctx = SkillContext(user_input="x")
        result = registry.execute("nonexistent", ctx)

        assert result.success is False
        assert "not found" in result.content.lower()

    def test_disabled_skill(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="old",
            description="Old skill",
            skill_type=SkillType.PROMPT,
            prompt_template="x",
            enabled=False,
        )
        registry.register(skill)

        ctx = SkillContext(user_input="x")
        result = registry.execute("old", ctx)

        assert result.success is False
        assert "disabled" in result.content.lower()