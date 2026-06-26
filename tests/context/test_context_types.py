"""测试上下文管理数据模型。"""

from __future__ import annotations

from kocor.context.budget import TokenBudget
from kocor.context.types import (
    AgentContext,
    ContextStrategy,
    MemoryItem,
    SummaryNode,
)
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class TestTokenBudget:
    """测试 TokenBudget 数据模型。"""

    def test_default_values(self):
        budget = TokenBudget()
        assert budget.limit == 200_000
        assert budget.used_prompt == 0
        assert budget.threshold_summary == 0.70
        assert budget.threshold_truncate == 0.90

    def test_remaining_when_unused(self):
        budget = TokenBudget(limit=100_000)
        assert budget.remaining == 100_000

    def test_remaining_deducts_used(self):
        budget = TokenBudget(limit=100_000, used_prompt=30_000)
        assert budget.remaining == 70_000

    def test_usage_ratio_zero(self):
        budget = TokenBudget()
        assert budget.usage_ratio == 0.0

    def test_usage_ratio_half(self):
        budget = TokenBudget(limit=100_000, used_prompt=50_000)
        assert budget.usage_ratio == 0.5

    def test_usage_ratio_full(self):
        budget = TokenBudget(limit=100_000, used_prompt=100_000)
        assert budget.usage_ratio == 1.0

    def test_should_not_summarize_below_threshold(self):
        budget = TokenBudget(limit=100_000, used_prompt=60_000)
        assert budget.should_summarize() is False

    def test_should_summarize_at_threshold(self):
        budget = TokenBudget(limit=100_000, used_prompt=70_000)
        assert budget.should_summarize() is True

    def test_should_summarize_above_threshold(self):
        budget = TokenBudget(limit=100_000, used_prompt=85_000)
        assert budget.should_summarize() is True

    def test_should_not_truncate_below_threshold(self):
        budget = TokenBudget(limit=100_000, used_prompt=80_000)
        assert budget.should_truncate() is False

    def test_should_truncate_at_threshold(self):
        budget = TokenBudget(limit=100_000, used_prompt=90_000)
        assert budget.should_truncate() is True

    def test_should_truncate_above_threshold(self):
        budget = TokenBudget(limit=100_000, used_prompt=95_000)
        assert budget.should_truncate() is True

    def test_zero_limit_does_not_divide_by_zero(self):
        budget = TokenBudget(limit=0)
        assert budget.usage_ratio == 0.0
        assert budget.should_summarize() is False
        assert budget.should_truncate() is False


class TestMemoryItem:
    """测试 MemoryItem 数据模型。"""

    def test_minimal_construction(self):
        item = MemoryItem(
            name="user-name",
            description="用户的名称",
            content="用户名: 张三",
            memory_type="user",
        )
        assert item.name == "user-name"
        assert item.description == "用户的名称"
        assert item.content == "用户名: 张三"
        assert item.memory_type == "user"
        assert item.created_at == ""
        assert item.updated_at == ""

    def test_all_fields(self):
        item = MemoryItem(
            name="test-item",
            description="测试",
            content="测试内容",
            memory_type="reference",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-06-20T00:00:00",
        )
        assert item.name == "test-item"
        assert item.memory_type == "reference"
        assert item.created_at == "2026-01-01T00:00:00"


class TestSummaryNode:
    """测试 SummaryNode 数据模型。"""

    def test_defaults(self):
        node = SummaryNode(
            summary="一段摘要",
            message_count=5,
            token_count=100,
            original_start=0,
            original_end=5,
        )
        assert node.summary == "一段摘要"
        assert node.message_count == 5
        assert node.token_count == 100
        assert node.original_end == 5


class TestAgentContext:
    """测试 AgentContext 数据模型。"""

    def test_default_construction(self):
        ctx = AgentContext(
            system_content="你是助手\n\n---\n\n项目指令",
            tool_definitions=[],
            session_messages=[],
        )
        assert ctx.system_content == "你是助手\n\n---\n\n项目指令"
        assert ctx.session_memory == {}
        assert ctx.token_budget.limit == 200_000

    def test_with_messages(self):
        msgs = [Message(role="system", content="hi")]
        ctx = AgentContext(
            system_content="",
            tool_definitions=[],
            session_messages=msgs,
        )
        assert len(ctx.session_messages) == 1
        assert ctx.session_messages[0].content == "hi"

    def test_with_tool_definitions(self):
        tools = [ToolDefinition(name="test", description="测试工具", parameters={})]
        ctx = AgentContext(
            system_content="",
            tool_definitions=tools,
            session_messages=[],
        )
        assert len(ctx.tool_definitions) == 1
        assert ctx.tool_definitions[0].name == "test"


class TestContextStrategy:
    """测试 ContextStrategy 枚举。"""

    def test_enum_values(self):
        assert ContextStrategy.DEFAULT.value == "default"
        assert ContextStrategy.SLIDING_WINDOW.value == "sliding"
        assert ContextStrategy.AGGRESSIVE.value == "aggressive"

    def test_enum_members(self):
        assert len(ContextStrategy) == 3
