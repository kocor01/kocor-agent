"""LLM 客户端 _normalize_tools 缓存测试。"""

from __future__ import annotations

from kocor.llm_provider.llm_client import BaseLLMClient
from kocor.llm_provider.message import Message
from kocor.tools.definitions import ToolDefinition


class CacheTestClient(BaseLLMClient):
    """用于测试工具缓存的假客户端。"""

    @property
    def provider(self) -> str:
        return "test"

    def _create_client(self):
        return object()

    def _normalize_in(self, messages):
        return [{"role": m.role, "content": m.content} for m in messages]

    def _normalize_out(self, response, usage=None):
        return Message(role="assistant", content="ok")

    def _normalize_tool(self, tool):
        return {"name": tool.name, "description": tool.description}

    def _api_generate(self, messages, tools, max_tokens, temperature, system=None):
        return None

    def _api_stream(self, messages, tools, max_tokens, temperature, system=None):
        return iter([])


class TestToolCache:
    def test_cache_hit_returns_same_object(self):
        """相同 tools 列表应返回缓存的对象。"""
        client = CacheTestClient()
        tools = [
            ToolDefinition(name="read", description="read file", parameters={"type": "object"}),
            ToolDefinition(name="write", description="write file", parameters={"type": "object"}),
        ]
        first = client._normalize_tools(tools)
        second = client._normalize_tools(tools)
        assert first is second  # 同一对象引用

    def test_different_tools_different_cache(self):
        """不同 tools 列表应使用不同缓存。"""
        client = CacheTestClient()
        tools_a = [ToolDefinition(name="read", description="read file", parameters={"type": "object"})]
        tools_b = [ToolDefinition(name="write", description="write file", parameters={"type": "object"})]
        result_a = client._normalize_tools(tools_a)
        result_b = client._normalize_tools(tools_b)
        assert result_a is not result_b
        assert result_a[0]["name"] == "read"
        assert result_b[0]["name"] == "write"

    def test_none_clears_cache(self):
        """传入 None 应清除缓存。"""
        client = CacheTestClient()
        tools = [ToolDefinition(name="read", description="read file", parameters={"type": "object"})]
        first = client._normalize_tools(tools)
        client._normalize_tools(None)  # 清除缓存
        second = client._normalize_tools(tools)
        assert first is not second  # 缓存被清除，重新创建

    def test_empty_tools_clears_cache(self):
        """传入空列表应清除缓存。"""
        client = CacheTestClient()
        tools = [ToolDefinition(name="read", description="read file", parameters={"type": "object"})]
        first = client._normalize_tools(tools)
        client._normalize_tools([])  # 空列表清除缓存
        second = client._normalize_tools(tools)
        assert first is not second