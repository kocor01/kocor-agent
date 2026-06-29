"""IterationBudget 测试。"""

import os

from kocor.config import Config
from kocor.harness.budget import IterationBudget


class TestIterationBudget:
    def setup_method(self):
        Config.reset()
        self._saved = {}
        for key in ["KOCOR_MAX_ITERATIONS"]:
            self._saved[key] = os.environ.pop(key, None)

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val

    def test_default_values(self):
        budget = IterationBudget()
        assert budget.used_iterations == 0
        assert budget.max_iterations == 20

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