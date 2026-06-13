"""Anthropic LLM 客户端实现。

使用 anthropic SDK，负责内部 Message 格式与 Anthropic API 格式之间的转换。
"""

from __future__ import annotations

import json
import os

from anthropic import Anthropic

from kocor.config import LLMConfig
from kocor.llm_client import LLMClient, ToolDefinition
from kocor.message import FunctionCall, Message, ToolCall


class AnthropicClient(LLMClient):
    """Anthropic LLM 客户端。

    Attributes:
        config: LLM 配置
        _model: 模型名称（从环境变量读取）
        _base_url: 兼容端点（从环境变量读取）
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = os.environ.get("ANTHROPIC_MODEL", "Opus 4.8")
        self._base_url = os.environ.get("ANTHROPIC_BASE_URL") or None

    @property
    def provider(self) -> str:
        return "anthropic"

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Message:
        """调用 Anthropic API 生成响应。

        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成长度
            temperature: 采样温度

        Returns:
            Message: 响应消息
        """
        client = Anthropic(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        # 提取 system 消息（Anthropic 用顶层参数）
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                filtered_messages.append(msg)

        anthropic_messages = self._normalize_in(filtered_messages)
        anthropic_tools = [self._normalize_tool(t) for t in tools] if tools else None

        # 调用 API
        response = client.messages.create(
            model=self._model,
            system=system_content or None,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=anthropic_tools,
        )

        # Anthropic 格式 → 内部格式
        return self._normalize_out(response)

    def _normalize_in(self, messages: list[Message]) -> list[dict]:
        """内部消息格式 → Anthropic 消息格式"""
        result = []
        for msg in messages:
            match msg.role:
                case "user":
                    result.append({"role": "user", "content": msg.content})
                case "assistant":
                    if msg.tool_calls:
                        content_blocks = []
                        for tc in msg.tool_calls:
                            # 如果有文本内容，先加 text block
                            if msg.content:
                                content_blocks.append({"type": "text", "text": msg.content})
                            # 加 tool_use block
                            try:
                                input_dict = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                input_dict = {}
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": input_dict,
                            })
                        result.append({"role": "assistant", "content": content_blocks})
                    else:
                        result.append({"role": "assistant", "content": msg.content})
                case "tool":
                    # Anthropic 中 tool 结果是 user 角色的 tool_result block
                    result.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    })
        return result

    def _normalize_out(self, response) -> Message:
        """Anthropic response 格式 → 内部消息格式"""
        content_blocks = response.content

        # 检查是否有工具调用
        tool_blocks = [b for b in content_blocks if b.type == "tool_use"]
        if tool_blocks:
            tool_calls = []
            for block in tool_blocks:
                input_data = block.input
                # dict → JSON 字符串
                if isinstance(input_data, dict):
                    arguments = json.dumps(input_data)
                else:
                    arguments = str(input_data)
                tool_calls.append(ToolCall(
                    id=block.id,
                    function=FunctionCall(
                        name=block.name,
                        arguments=arguments,
                    ),
                ))
            return Message(
                role="assistant",
                content="",
                tool_calls=tool_calls,
            )

        # 纯文本响应
        text_blocks = [b for b in content_blocks if b.type == "text"]
        content = " ".join(b.text for b in text_blocks) if text_blocks else ""
        return Message(role="assistant", content=content)

    def _normalize_tool(self, tool: ToolDefinition) -> dict:
        """内部工具定义 → Anthropic 工具格式"""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
