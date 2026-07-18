"""后台记忆审查。

每 N 轮触发一次，由独立 LLM 回顾会话并判断是否值得长期记忆。
"""

from __future__ import annotations

import json

from kocor.config import Config
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message
from kocor.memory.store import MemoryStore
from kocor.tools.definitions import ToolDefinition
from kocor.tools.toolsets.memory_tool import MemoryTool

MEMORY_REVIEW_PROMPT = """你正在回顾一段对话，判断是否有值得长期记忆的内容。

关注：
1. 用户是否透露了关于自己的信息——身份、偏好、个人细节？
2. 用户是否表达了对 Agent 行为方式、工作风格的期望？

如有值得记忆的内容，使用 memory 工具保存。
如无值得保存的内容，回复 'Nothing to save.' 并停止。"""


class BackgroundReviewer:
    """后台记忆审查器。"""

    def __init__(self, llm: LLMClient, store: MemoryStore):
        self.llm = llm
        self.store = store

    def review(self, messages: list[Message]) -> None:
        memory_tool = ToolDefinition(
            name=MemoryTool.NAME,
            description=MemoryTool.DESCRIPTION,
            parameters=MemoryTool.PARAMETERS,
            safety_level=MemoryTool.SAFETY_LEVEL,
        )

        # 构建系统提示：附上当前已保存的记忆，供 LLM 判断增量
        current_memory = self.store.format_for_system_prompt()
        system_prompt = MEMORY_REVIEW_PROMPT
        if current_memory:
            system_prompt += (
                "\n\n当前已保存的记忆（请在此基础之上判断是否需要新增/更新）：\n"
                + current_memory
            )

        review_msgs = [
            Message(role="system", content=system_prompt),
        ]
        # 取最近 nudge_interval 条消息供审查（配置为 0 时取全部）
        window = Config.load().nudge_interval or len(messages)
        recent = messages[-window:] if len(messages) > window else messages
        review_msgs.extend(recent)

        response = self.llm.generate(review_msgs, tools=[memory_tool])
        if not response.tool_calls:
            return

        for tc in response.tool_calls:
            if tc.function.name == "memory":
                try:
                    ops = json.loads(tc.function.arguments).get("operations", [])
                except json.JSONDecodeError:
                    continue
                if not ops:
                    continue
                MemoryTool.handler(store=self.store, operations=ops)
