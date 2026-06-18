"""测试技能暴露为 ToolRegistry 工具。"""

from kocor.llm_provider.message import FunctionCall, ToolCall
from kocor.skill.models import InvokeStrategy, SkillDefinition, SkillType
from kocor.skill.registry import SkillRegistry
from kocor.tool_registry import ToolRegistry


class TestRegisterAsTools:
    """测试 register_as_tools()"""

    def test_llm_skill_registered_as_tool(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        skill = SkillDefinition(
            name="format",
            description="Format code",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="Format this: {user_input}",
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools(tool_registry)

        defs = tool_registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].name == "skill_format"
        assert "Format code" in defs[0].description

    def test_both_skill_registered_as_tool(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        skill = SkillDefinition(
            name="review",
            description="Review code",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.BOTH,
            prompt_template="Review: {user_input}",
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools(tool_registry)

        assert len(tool_registry.get_definitions()) == 1

    def test_slash_skill_not_registered(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        skill = SkillDefinition(
            name="deploy",
            description="Deploy app",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools(tool_registry)

        assert len(tool_registry.get_definitions()) == 0

    def test_disabled_skill_not_registered(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        skill = SkillDefinition(
            name="old",
            description="Old skill",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="x",
            enabled=False,
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools(tool_registry)

        assert len(tool_registry.get_definitions()) == 0

    def test_mixed_skills(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        skills = [
            SkillDefinition(name="a", description="A", skill_type=SkillType.PROMPT,
                            invoke_strategy=InvokeStrategy.BOTH, prompt_template="a"),
            SkillDefinition(name="b", description="B", skill_type=SkillType.PROMPT,
                            invoke_strategy=InvokeStrategy.SLASH, prompt_template="b"),
            SkillDefinition(name="c", description="C", skill_type=SkillType.PROMPT,
                            invoke_strategy=InvokeStrategy.LLM, prompt_template="c"),
        ]
        for s in skills:
            skill_registry.register(s)
        skill_registry.register_as_tools(tool_registry)

        names = {d.name for d in tool_registry.get_definitions()}
        assert names == {"skill_a", "skill_c"}

    def test_tool_execution_wraps_skill(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        def greet_handler(user_input: str) -> str:
            return f"Hi, {user_input}!"

        skill = SkillDefinition(
            name="greet",
            description="Greet someone",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.BOTH,
            handler=greet_handler,
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools(tool_registry)

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="skill_greet", arguments='{"user_input": "world"}'),
        )
        result = tool_registry.execute(tool_call)
        assert result.content == "Hi, world!"

    def test_tool_prompt_skill_execution(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        skill = SkillDefinition(
            name="review",
            description="Review code",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="Review this: {user_input}",
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools(tool_registry)

        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="skill_review", arguments='{"user_input": "def foo(): pass"}'),
        )
        result = tool_registry.execute(tool_call)
        assert result.content == "Review this: def foo(): pass"

    def test_no_tool_registry_provided_uses_constructor(self):
        tr = ToolRegistry()
        skill_registry = SkillRegistry(tool_registry=tr)

        skill = SkillDefinition(
            name="test",
            description="Test",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="Test: {user_input}",
        )
        skill_registry.register(skill)
        skill_registry.register_as_tools()

        assert len(tr.get_definitions()) == 1

    def test_no_tool_registry_at_all(self):
        skill_registry = SkillRegistry()

        skill = SkillDefinition(
            name="test",
            description="Test",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="x",
        )
        skill_registry.register(skill)
        # Should not raise
        skill_registry.register_as_tools()