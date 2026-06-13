"""Agent 核心。

自主 Agent 核心循环：query LLM → call tool → observe → loop until final answer。
"""

from __future__ import annotations

from kocor.llm_client import LLMClient, ToolDefinition
from kocor.message import Message
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
