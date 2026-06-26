"""测试 ContextBuilder。"""

from __future__ import annotations

from kocor.context.builder import ContextBuilder
from kocor.context.types import AgentContext
from kocor.tools.definitions import ToolDefinition


# 简单的 ToolRegistry stub（不依赖真实实现）
class ToolRegistryStub:
    def __init__(self, tools: list[ToolDefinition] | None = None):
        self._tools = tools or []

    def get_definitions(self) -> list[ToolDefinition]:
        return self._tools


class TestContextBuilder:
    """测试 ContextBuilder 基本功能。"""

    def test_default_construction(self):
        """默认构造应成功。"""
        tools = ToolRegistryStub()
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=tools,
        )
        assert builder.identity_prompt == "你是 Kocor"

    def test_build_system_prompt_contains_identity(self):
        """system prompt 应包含身份提示。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor 助手",
            tools=ToolRegistryStub(),
        )
        prompt = builder.build_system_prompt()
        assert "你是 Kocor 助手" in prompt

    def test_build_system_prompt_includes_env_info(self):
        """system prompt 应包含环境信息。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        prompt = builder.build_system_prompt()
        assert "当前工作目录" in prompt

    def test_build_system_prompt_with_custom_instructions(self):
        """项目指令文件不存在时不报错，只跳过 L2 层。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        prompt = builder.build_system_prompt()
        assert "你是 Kocor" in prompt

    def test_build_context_returns_agent_context(self):
        """build_context 应返回 AgentContext。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        ctx = builder.build_context(user_input="你好", session_history=[])
        assert isinstance(ctx, AgentContext)
        assert ctx.token_budget is not None

    def test_build_context_includes_user_input(self):
        """build_context 应包含用户输入作为最后一条消息。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        ctx = builder.build_context(user_input="帮我读文件", session_history=[])
        messages = ctx.session_messages
        assert messages[-1].role == "user"
        assert messages[-1].content == "帮我读文件"

    def test_build_context_includes_system_prompt(self):
        """build_context 的第一条消息应为 system prompt。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor 助手",
            tools=ToolRegistryStub(),
        )
        ctx = builder.build_context(user_input="你好", session_history=[])
        assert ctx.session_messages[0].role == "system"
        assert len(ctx.session_messages[0].content) > 0

    def test_build_context_with_history(self):
        """会话历史应包含在消息列表中。"""
        from kocor.llm_provider.message import Message

        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        history = [
            Message(role="user", content="第一轮"),
            Message(role="assistant", content="回答1"),
        ]
        ctx = builder.build_context(user_input="第二轮", session_history=history)
        messages = ctx.session_messages
        # system + 历史(2) + 当前用户输入
        assert len(messages) == 4
        assert messages[1].role == "user"
        assert messages[1].content == "第一轮"
        assert messages[3].role == "user"
        assert messages[3].content == "第二轮"

    def test_build_context_empty_history(self):
        """空历史应只有 system + user。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        ctx = builder.build_context(user_input="你好", session_history=[])
        assert len(ctx.session_messages) == 2
        assert ctx.session_messages[0].role == "system"
        assert ctx.session_messages[1].role == "user"

    def test_context_contains_tool_definitions(self):
        """context 应包含工具定义。"""
        tools = [
            ToolDefinition(name="read_file", description="读文件", parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }),
        ]
        stub = ToolRegistryStub(tools)
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=stub,
        )
        ctx = builder.build_context(user_input="读文件", session_history=[])
        assert len(ctx.tool_definitions) == 1
        assert ctx.tool_definitions[0].name == "read_file"

    def test_context_has_environment_info(self):
        """环境信息应包含在 system prompt 文本中。"""
        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
        )
        ctx = builder.build_context(user_input="你好", session_history=[])
        system_msg = ctx.session_messages[0].content
        assert "当前工作目录" in system_msg

    # ── 与 MemoryManager 集成 ──────────────────────────

    def test_memories_in_system_prompt(self):
        """记忆应注入到 system prompt 中。"""
        import tempfile
        from kocor.context.memory import MemoryManager
        from kocor.context.types import MemoryItem

        mem_dir = tempfile.mkdtemp()
        memory = MemoryManager(memory_dir=mem_dir)
        memory.save(MemoryItem(
            name="user-name", description="用户名", content="用户: 张三", memory_type="user",
        ))

        builder = ContextBuilder(
            identity_prompt="你是 Kocor",
            tools=ToolRegistryStub(),
            memory=memory,
        )
        prompt = builder.build_system_prompt()
        assert "已记录的信息" in prompt
        assert "用户: 张三" in prompt