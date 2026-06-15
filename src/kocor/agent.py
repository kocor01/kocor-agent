"""Agent 核心。

自主 Agent 核心循环：query LLM → call tool → observe → loop until final answer。
"""

from __future__ import annotations

from typing import Iterator

from kocor.llm_client import LLMClient, ToolDefinition
from kocor.message import Message, StreamChunk, ToolCall, ToolResult
from kocor.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """\
你是一个名为 Kocor 的 AI 助手，擅长通过调用工具来完成任务。

你的能力:
- 读取和写入文件
- 在沙盒中执行 Python 代码

工作原则:
1. 理解用户意图后，选择合适的工具完成任务
2. 如果需要多次操作，逐步执行，每次只做一个合理的操作
3. 工具执行后，根据结果决定下一步
4. 任务完成后，给出清晰简洁的总结
5. 如果不确定，可以向用户提问（通过回复纯文本）\
"""


class Agent:
    """自主 Agent 核心。

    Attributes:
        llm: LLM 客户端
        tools: 工具注册表
        system_prompt: 系统提示词
        max_iterations: 最大迭代次数
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_iterations = max_iterations

    def run(self, user_input: str) -> str:
        """执行一次完整的 Agent 循环。

        Args:
            user_input: 用户输入

        Returns:
            最终文本答案
        """
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]

        for _ in range(self.max_iterations):
            # 1. 调用 LLM
            response = self.llm.generate(
                messages,
                tools=self.tools.get_definitions(),
            )
            messages.append(response)

            # 2. 检查是否有工具调用
            if not response.tool_calls:
                return response.content  # 最终答案

            # 3. 执行工具
            for tool_call in response.tool_calls:
                result = self.tools.execute(tool_call)
                messages.append(Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                ))

       # 超时
        return f"Agent 在 {self.max_iterations} 次迭代后仍未完成，可能任务过于复杂。"

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 Agent 循环。

        Args:
            user_input: 用户输入

        Yields:
            StreamChunk: 流式数据块
        """
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]

        for _ in range(self.max_iterations):
            # 1. 流式调用 LLM
            accumulated_content = ""
            accumulated_tool_calls: list[ToolCall] = []

            for chunk in self.llm.stream(
                messages,
                tools=self.tools.get_definitions(),
            ):
                accumulated_content += chunk.content
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        # 避免重复添加同一 tool_call
                        if not any(t.id == tc.id for t in accumulated_tool_calls):
                            accumulated_tool_calls.append(tc)
                yield chunk

                # 收到 is_final 后处理
                if chunk.is_final:
                    # 将完整的 assistant 消息追加到 messages
                    if accumulated_tool_calls:
                        messages.append(Message(
                            role="assistant",
                            content=accumulated_content,
                            tool_calls=accumulated_tool_calls,
                        ))
                    else:
                        messages.append(Message(
                            role="assistant",
                            content=accumulated_content,
                        ))

                    # 2. 检查是否有工具调用
                    if not accumulated_tool_calls:
                        # 最终答案，结束循环
                        return

                    # 3. 执行工具（阻塞，不 yield），执行后 yield 结果块
                    for tool_call in accumulated_tool_calls:
                        result = self.tools.execute(tool_call)
                        messages.append(Message(
                            role="tool",
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                        ))
                        yield StreamChunk(tool_result=result, is_final=True)

        # 超时
        yield StreamChunk(
            content=f"Agent 在 {self.max_iterations} 次迭代后仍未完成，可能任务过于复杂。",
            is_final=True,
        )
