"""后台记忆审查。

每 N 轮触发一次，由独立 LLM 回顾会话并判断是否值得长期记忆。
"""

from __future__ import annotations

import json

from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message
from kocor.memory.store import MemoryStore
from kocor.memory.types import MemoryTarget
from kocor.tools.definitions import ToolDefinition
from kocor.tools.permission import PermissionManager

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
            name="memory",
            description=("保存长期记忆。你可以通过 operations 数组一次性 add/replace/remove。"
                         'add: {"action":"add", "target":"memory|user", "content":"..."}'),
            parameters={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"enum": ["add", "replace", "remove"]},
                                "target": {"enum": ["memory", "user"]},
                                "content": {"type": "string"},
                                "old_substring": {"type": "string"},
                            },
                        },
                    },
                },
            },
            safety_level=PermissionManager.SAFETY_SAFE,
        )

        review_msgs = [
            Message(role="system", content=MEMORY_REVIEW_PROMPT),
        ]
        recent = messages[-6:] if len(messages) > 6 else messages
        review_msgs.extend(recent)

        response = self.llm.generate(review_msgs, tools=[memory_tool])
        if not response.tool_calls:
            return

        for tc in response.tool_calls:
            if tc.function.name == "memory":
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    continue
                ops = args.get("operations", [])
                if not ops:
                    continue
                self._handle_memory_ops(ops)

    def _handle_memory_ops(self, ops: list[dict]) -> None:
        for op in ops:
            action = op.get("action")
            try:
                target = MemoryTarget(op.get("target", "memory"))
            except ValueError:
                continue

            if action == "add":
                content = op.get("content", "")
                if content:
                    self.store.add(target, content)
            elif action == "replace":
                self.store.replace(target, op.get("old_substring", ""), op.get("content", ""))
            elif action == "remove":
                self.store.remove(target, op.get("old_substring", ""))
