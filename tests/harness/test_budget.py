"""IterationBudget 测试。"""

import os

from kocor.config import Config
from kocor.harness.budget import IterationBudget


class TestIterationBudget:
    def setup_method(self):
        Config.reset()
        for key in list(os.environ):
            if key.startswith("KOCOR_"):
                os.environ.pop(key, None)

    def test_default_values(self):
        budget = IterationBudget()
        assert budget.used_iterations == 0
        # max_iterations 从 Config.load().max_iterations 加载（默认 20）
        assert budget.max_iterations == Config.load().max_iterations

    def test_not_exhausted_by_default(self):
        budget = IterationBudget()
        assert not budget.exhausted

    def test_exhausted_by_iterations(self):
        budget = IterationBudget(max_iterations=3)
        budget.used_iterations = 3
        assert budget.exhausted

    def test_remaining_iterations(self):
        budget = IterationBudget(max_iterations=10)
        budget.used_iterations = 3
        assert budget.remaining_iterations == 7

    def test_remaining_iterations_when_exhausted(self):
        budget = IterationBudget(max_iterations=5)
        budget.used_iterations = 5
        assert budget.remaining_iterations == 0

    def test_custom_values(self):
        budget = IterationBudget(max_iterations=5)
        assert budget.max_iterations == 5

    def test_reset(self):
        budget = IterationBudget(max_iterations=10)
        budget.used_iterations = 5
        budget.reset()

        assert budget.used_iterations == 0