"""Agent 循环的迭代预算追踪。"""

import time
from dataclasses import dataclass

from kocor.config import config_get


@dataclass
class IterationBudget:
    """追踪 Agent 迭代中的资源消耗。

    监控三个维度：迭代次数、Token 用量和运行时长。
    循环控制器在每次迭代后检查 `exhausted`。
    """

    iterations_used: int = 0       # 已用迭代次数
    iterations_limit: int = config_get("max_iterations")  # 迭代次数上限

    tokens_prompt: int = 0         # 已用 Prompt Token 数
    tokens_completion: int = 0     # 已用 Completion Token 数
    tokens_limit: int = config_get("context_max_tokens")  # Token 总上限

    time_start: float = 0.0        # 起始时间戳（unix）
    time_elapsed: float = 0.0      # 已用时长（秒）
    time_limit: float = config_get("timeout")  # 运行时长上限（秒）

    def __post_init__(self):
        if self.time_start == 0.0:
            self.time_start = time.time()

    @property
    def exhausted(self) -> bool:
        """如果任一预算维度已超限，返回 True。"""
        self.time_elapsed = time.time() - self.time_start
        return (
            self.iterations_used >= self.iterations_limit
            or self.tokens_prompt >= self.tokens_limit
            or self.time_elapsed >= self.time_limit
        )

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.iterations_limit - self.iterations_used)

    def reset(self) -> None:
        """重置所有计数器（用于跨会话复用）。"""
        self.iterations_used = 0
        self.tokens_prompt = 0
        self.tokens_completion = 0
        self.time_start = time.time()
        self.time_elapsed = 0.0