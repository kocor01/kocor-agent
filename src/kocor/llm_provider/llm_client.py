"""LLM 客户端抽象层。

提供统一的 LLMClient 接口（Protocol），具体实现由 OpenAI/Anthropic SDK 完成。
BaseLLMClient 是抽象基类，提供 generate/stream 模板方法，减少子类重复代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Protocol

import httpx

from kocor.config import Config
from kocor.llm_provider.exceptions import LLMConnectionError, LLMTimeoutError
from kocor.llm_provider.message import Message, StreamChunk, Usage
from kocor.tools.definitions import ToolDefinition


class LLMClient(Protocol):
    """LLM 客户端抽象接口。

    所有 provider 实现必须遵循此接口。
    """

    @property
    def provider(self) -> str:
        """返回 provider 名称: 'openai' | 'anthropic'"""
        ...

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Message:
        """生成响应。

        Args:
            messages: 消息列表（含历史）
            tools: 可用工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Returns:
            Message:
            - 纯文本: Message(role="assistant", content="...")
            - 工具调用: Message(role="assistant", content="", tool_calls=[...])
        """
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式生成响应。

        Args:
            messages: 消息列表（含历史）
            tools: 可用工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Yields:
            StreamChunk: 流式数据块
        """
        ...


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类。

    子类只需实现格式转换方法（_normalize_in/_normalize_out/_normalize_tool）、
    API 调用方法（_api_generate/_api_stream）和 _create_client()。
    公共逻辑（max_tokens 默认值、超时处理、模板方法）由基类提供。

    子类可重写 _prepare_messages() 以预处理消息（如提取 system 消息）。
    """

    def __init__(self):
        self.config = Config.load()
        self._client = self._create_client()
        # 工具定义缓存：(id(tools), 转换结果)
        # 工具列表在会话期间不变（ToolManager 持有一份引用），
        # 用 id(tools) 作缓存键精确且零开销。
        self._tool_cache: tuple[int, list[dict] | None] | None = None

    @property
    @abstractmethod
    def provider(self) -> str:
        """返回 provider 名称。"""
        ...

    @abstractmethod
    def _create_client(self):
        """创建 SDK 客户端实例。"""
        ...

    @abstractmethod
    def _normalize_in(self, messages: list[Message]) -> list[dict]:
        """内部消息格式 → Provider 消息格式。"""
        ...

    @abstractmethod
    def _normalize_out(self, response, usage: Usage | None = None) -> Message:
        """Provider 响应格式 → 内部消息格式。"""
        ...

    @abstractmethod
    def _normalize_tool(self, tool: ToolDefinition) -> dict:
        """内部工具定义 → Provider 工具格式。"""
        ...

    @abstractmethod
    def _api_generate(self, messages_data, tools_data, max_tokens, temperature, system=None):
        """调用 Provider API 生成响应（非流式）。

        Args:
            messages_data: _normalize_in 处理后的消息数据
            tools_data: _normalize_tool 处理后的工具数据
            max_tokens: 最大生成长度
            temperature: 采样温度
            system: _prepare_messages 提取的 system 数据（如无则为 None）

        Returns:
            原始 SDK 响应对象，传给 _normalize_out
        """
        ...

    @abstractmethod
    def _api_stream(self, messages_data, tools_data, max_tokens, temperature, system=None):
        """调用 Provider API 流式生成响应。

        Args:
            同 _api_generate

        Yields:
            原始流式事件，由子类自行处理为 StreamChunk
        """
        ...

    def _prepare_messages(self, messages: list[Message]) -> tuple:
        """预处理消息。子类可重写，如提取 system 消息。

        Returns:
            (system_data, messages_for_normalize)
            system_data: 传给 _api_generate/_api_stream 的 system 参数
            messages_for_normalize: 传给 _normalize_in 的消息列表
        """
        return None, messages

    def _normalize_tools(
        self,
        tools: list[ToolDefinition] | None,
    ) -> list[dict] | None:
        """转换工具定义，带缓存。

        工具列表在会话期间不变（ToolManager 持有一份引用），
        用 id(tools) 作缓存键。tools 为 None 时清除缓存。

        Args:
            tools: 工具定义列表或 None

        Returns:
            转换后的工具数据列表或 None
        """
        if not tools:
            self._tool_cache = None
            return None

        cache_key = id(tools)
        if self._tool_cache is not None and self._tool_cache[0] == cache_key:
            return self._tool_cache[1]

        result = [self._normalize_tool(t) for t in tools]
        self._tool_cache = (cache_key, result)
        return result

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Message:
        """生成响应。

        模板方法：_prepare_messages → _normalize_in → _normalize_tools → _api_generate → _normalize_out
        """
        # 未指定 max_tokens 时使用配置值，允许调用方传 None 表示"使用默认"
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        system_data, filtered = self._prepare_messages(messages)
        input_data = self._normalize_in(filtered)
        tool_data = self._normalize_tools(tools)
        try:
            response = self._api_generate(input_data, tool_data, actual_max_tokens, temperature, system=system_data)
            return self._normalize_out(response)
        except httpx.TimeoutException:
            raise LLMTimeoutError(provider=self.provider, timeout_seconds=self.config.tool_timeout)
        except httpx.ConnectError:
            raise LLMConnectionError(provider=self.provider)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式生成响应。

        模板方法：_prepare_messages → _normalize_in → _normalize_tools → _api_stream
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        system_data, filtered = self._prepare_messages(messages)
        input_data = self._normalize_in(filtered)
        tool_data = self._normalize_tools(tools)
        try:
            yield from self._api_stream(input_data, tool_data, actual_max_tokens, temperature, system=system_data)
        except httpx.TimeoutException:
            raise LLMTimeoutError(provider=self.provider, timeout_seconds=self.config.tool_timeout)
        except httpx.ConnectError:
            raise LLMConnectionError(provider=self.provider)