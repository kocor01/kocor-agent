"""测试 Skill 数据模型。"""

from kocor.skill.models import (
    InvokeStrategy,
    SkillContext,
    SkillDefinition,
    SkillResult,
    SkillType,
)
from kocor.tools.tool_manager import ToolManager


class TestSkillType:
    """测试 SkillType 枚举"""

    def test_prompt_value(self):
        assert SkillType.PROMPT.value == "prompt"

    def test_code_value(self):
        assert SkillType.CODE.value == "code"


class TestInvokeStrategy:
    """测试 InvokeStrategy 枚举"""

    def test_slash_value(self):
        assert InvokeStrategy.SLASH.value == "slash"

    def test_llm_value(self):
        assert InvokeStrategy.LLM.value == "llm"

    def test_both_value(self):
        assert InvokeStrategy.BOTH.value == "both"


class TestSkillDefinition:
    """测试 SkillDefinition 数据类"""

    def test_minimal_prompt_skill(self):
        skill = SkillDefinition(
            name="review",
            description="Review code",
            skill_type=SkillType.PROMPT,
            prompt_template="Review: {user_input}",
        )
        assert skill.name == "review"
        assert skill.description == "Review code"
        assert skill.skill_type == SkillType.PROMPT
        assert skill.invoke_strategy == InvokeStrategy.BOTH
        assert skill.prompt_template == "Review: {user_input}"
        assert skill.prompt_role == "user"
        assert skill.handler is None
        assert skill.parameters is None
        assert skill.category == "general"
        assert skill.enabled is True
        assert skill.version == "1.0.0"

    def test_code_skill_with_handler(self):
        def my_handler(user_input: str) -> str:
            return f"hello {user_input}"

        skill = SkillDefinition(
            name="greet",
            description="Greet someone",
            skill_type=SkillType.CODE,
            handler=my_handler,
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        assert skill.name == "greet"
        assert skill.skill_type == SkillType.CODE
        assert skill.handler is my_handler
        assert skill.parameters is not None

    def test_slash_only_skill(self):
        skill = SkillDefinition(
            name="deploy",
            description="Deploy app",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
        )
        assert skill.invoke_strategy == InvokeStrategy.SLASH

    def test_llm_only_skill(self):
        skill = SkillDefinition(
            name="format_code",
            description="Format code",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="Format this:\n{user_input}",
        )
        assert skill.invoke_strategy == InvokeStrategy.LLM

    def test_disabled_skill(self):
        skill = SkillDefinition(
            name="old_skill",
            description="An old skill",
            skill_type=SkillType.PROMPT,
            enabled=False,
        )
        assert skill.enabled is False

    def test_custom_category(self):
        skill = SkillDefinition(
            name="test",
            description="Test",
            skill_type=SkillType.PROMPT,
            category="development",
        )
        assert skill.category == "development"

    def test_custom_version_and_author(self):
        skill = SkillDefinition(
            name="test",
            description="Test",
            skill_type=SkillType.PROMPT,
            version="2.0.0",
            author="kocor",
        )
        assert skill.version == "2.0.0"
        assert skill.author == "kocor"

    def test_system_role_prompt(self):
        skill = SkillDefinition(
            name="system_skill",
            description="System skill",
            skill_type=SkillType.PROMPT,
            prompt_template="You are an expert.",
            prompt_role="system",
        )
        assert skill.prompt_role == "system"


class TestSkillContext:
    """测试 SkillContext 数据类"""

    def test_minimal_context(self):
        ctx = SkillContext(user_input="hello")
        assert ctx.user_input == "hello"
        assert ctx.tool_manager is None
        assert ctx.extra == {}

    def test_with_tool_manager(self):
        registry = ToolManager()
        ctx = SkillContext(user_input="hello", tool_manager=registry)
        assert ctx.tool_manager is registry

    def test_with_extra(self):
        ctx = SkillContext(user_input="hello", extra={"key": "value"})
        assert ctx.extra == {"key": "value"}


class TestSkillResult:
    """测试 SkillResult 数据类"""

    def test_success_result(self):
        result = SkillResult(content="done", skill_name="test")
        assert result.content == "done"
        assert result.skill_name == "test"
        assert result.success is True
        assert result.error is None

    def test_error_result(self):
        result = SkillResult(
            content="failed",
            skill_name="test",
            success=False,
            error="Something went wrong",
        )
        assert result.content == "failed"
        assert result.success is False
        assert result.error == "Something went wrong"
