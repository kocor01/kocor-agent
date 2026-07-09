"""Agent 循环的迭代预算追踪。"""

from dataclasses import dataclass

from kocor.config import Config


@dataclass
class IterationBudget:
    """追踪 Agent ReAct 循环的迭代次数。"""

    used_iterations: int = 0
    max_iterations: int = Config.load().max_iterations

    @property
    def exhausted(self) -> bool:
        return self.used_iterations >= self.max_iterations

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.used_iterations)

    def reset(self) -> None:
        self.used_iterations = 0