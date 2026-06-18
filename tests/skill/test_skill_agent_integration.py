"""测试 Agent 与 Skill 集成（slash 命令处理）。"""

from unittest.mock import MagicMock, patch

from kocor.agent import Agent
from kocor.llm_provider.message import Message, StreamChunk
from kocor.skill.models import InvokeStrategy, SkillDefinition, SkillType
from kocor.skill.registry import SkillRegistry
from kocor.tool_registry import ToolRegistry


def _make_mock_llm(content: str = "final answer"):
    """创建一个返回固定文本的 mock LLM 客户端。"""
    llm = MagicMock()
    llm.provider = "test"
    msg = Message(role="assistant", content=content)
    llm.generate.return_value = msg
    return llm


class TestSlashCommandRun:
    """测试 run() 中的 slash 命令处理"""

    def test_slash_unknown_skill(self):
        skill_registry = SkillRegistry()
        agent = Agent(
            llm=_make_mock_llm(),
            skills=skill_registry,
        )
        result = agent.run("/nonexistent")
        assert "Unknown skill" in result
        assert "nonexistent" in result

    def test_slash_code_skill(self):
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()

        def greet_handler(user_input: str) -> str:
            return f"Hello, {user_input}!"

        skill = SkillDefinition(
            name="greet",
            description="Greet",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
            handler=greet_handler,
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=_make_mock_llm(),
            tools=tool_registry,
            skills=skill_registry,
        )
        result = agent.run("/greet world")
        assert result == "Hello, world!"

    def test_slash_code_skill_no_args(self):
        skill_registry = SkillRegistry()

        def no_args_handler(user_input: str) -> str:
            return f"input: '{user_input}'"

        skill = SkillDefinition(
            name="ping",
            description="Ping",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
            handler=no_args_handler,
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=_make_mock_llm(),
            skills=skill_registry,
        )
        result = agent.run("/ping")
        assert result == "input: ''"

    def test_slash_prompt_skill(self):
        skill_registry = SkillRegistry()
        llm = _make_mock_llm("Reviewed!")

        skill = SkillDefinition(
            name="review",
            description="Review",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.BOTH,
            prompt_template="Review this: {user_input}",
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=llm,
            skills=skill_registry,
        )
        result = agent.run("/review def foo(): pass")
        assert result == "Reviewed!"
        # Verify the prompt was injected into the LLM call
        call_args = llm.generate.call_args
        messages = call_args[0][0]  # first positional arg
        assert any("Review this: def foo(): pass" in m.content for m in messages)

    def test_slash_llm_only_skill_rejected(self):
        skill_registry = SkillRegistry()

        skill = SkillDefinition(
            name="internal",
            description="Internal only",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.LLM,
            prompt_template="x",
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=_make_mock_llm(),
            skills=skill_registry,
        )
        result = agent.run("/internal")
        assert "cannot be invoked" in result.lower()

    def test_slash_for_disabled_skill(self):
        skill_registry = SkillRegistry()

        skill = SkillDefinition(
            name="old",
            description="Old",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
            handler=lambda user_input: "should not run",
            enabled=False,
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=_make_mock_llm(),
            skills=skill_registry,
        )
        result = agent.run("/old")
        assert "disabled" in result.lower() or "old" in result.lower()

    def test_slash_prompt_with_system_role(self):
        skill_registry = SkillRegistry()
        llm = _make_mock_llm("Done")

        skill = SkillDefinition(
            name="expert",
            description="Expert mode",
            skill_type=SkillType.PROMPT,
            invoke_strategy=InvokeStrategy.BOTH,
            prompt_template="You are an expert in {user_input}",
            prompt_role="system",
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=llm,
            skills=skill_registry,
        )
        result = agent.run("/expert Python")
        assert result == "Done"

    def test_regular_input_not_intercepted(self):
        skill_registry = SkillRegistry()
        llm = _make_mock_llm("normal answer")
        agent = Agent(
            llm=llm,
            skills=skill_registry,
        )
        result = agent.run("hello")
        assert result == "normal answer"


class TestSlashCommandStream:
    """测试 stream() 中的 slash 命令处理"""

    def test_slash_code_skill_stream(self):
        skill_registry = SkillRegistry()

        def greet_handler(user_input: str) -> str:
            return f"Hello, {user_input}!"

        skill = SkillDefinition(
            name="greet",
            description="Greet",
            skill_type=SkillType.CODE,
            invoke_strategy=InvokeStrategy.SLASH,
            handler=greet_handler,
        )
        skill_registry.register(skill)

        agent = Agent(
            llm=_make_mock_llm(),
            skills=skill_registry,
        )
        chunks = list(agent.stream("/greet world"))
        assert len(chunks) == 1
        assert chunks[0].content == "Hello, world!"
        assert chunks[0].is_final is True

    def test_slash_unknown_skill_stream(self):
        skill_registry = SkillRegistry()
        agent = Agent(
            llm=_make_mock_llm(),
            skills=skill_registry,
        )
        chunks = list(agent.stream("/nonexistent"))
        assert "Unknown skill" in chunks[0].content

    def test_regular_input_not_intercepted_stream(self):
        skill_registry = SkillRegistry()
        llm = MagicMock()
        llm.stream.return_value = iter([
            StreamChunk(content="answer", is_final=True),
        ])

        agent = Agent(
            llm=llm,
            skills=skill_registry,
        )
        chunks = list(agent.stream("hello"))
        assert any("answer" in c.content for c in chunks)


class TestNoSkills:
    """测试没有 skill registry 时的行为"""

    def test_run_without_skills(self):
        agent = Agent(llm=_make_mock_llm("ok"))
        result = agent.run("hello")
        assert result == "ok"

    def test_stream_without_skills(self):
        llm = MagicMock()
        llm.stream.return_value = iter([
            StreamChunk(content="result", is_final=True),
        ])
        agent = Agent(llm=llm)
        chunks = list(agent.stream("hello"))
        assert chunks[0].content == "result"