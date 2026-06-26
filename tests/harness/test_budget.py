"""IterationBudget 测试。"""

import os
import time

from kocor.config import Config
from kocor.harness.budget import IterationBudget


class TestIterationBudget:
    def setup_method(self):
        Config.reset()
        self._saved = {}
        for key in ["KOCOR_MAX_ITERATIONS", "KOCOR_TIMEOUT", "KOCOR_CONTEXT_MAX_TOKENS"]:
            self._saved[key] = os.environ.pop(key, None)

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val

    def test_default_values(self):
        budget = IterationBudget()
        assert budget.iterations_used == 0
        assert budget.iterations_limit == 20
        assert budget.tokens_prompt == 0
        assert budget.tokens_completion == 0
        assert budget.tokens_limit == 200_000

    def test_not_exhausted_by_default(self):
        budget = IterationBudget()
        assert not budget.exhausted

    def test_exhausted_by_iterations(self):
        budget = IterationBudget(iterations_limit=3)
        budget.iterations_used = 3
        assert budget.exhausted

    def test_exhausted_by_tokens(self):
        budget = IterationBudget(tokens_limit=1000)
        budget.tokens_prompt = 1000
        assert budget.exhausted

    def test_exhausted_by_time(self):
        budget = IterationBudget(time_limit=0.1)
        budget.time_start = time.time() - 1.0
        budget.time_elapsed = 0.2
        assert budget.exhausted

    def test_remaining_iterations(self):
        budget = IterationBudget(iterations_limit=10)
        budget.iterations_used = 3
        assert budget.remaining_iterations == 7

    def test_remaining_iterations_when_exhausted(self):
        budget = IterationBudget(iterations_limit=5)
        budget.iterations_used = 5
        assert budget.remaining_iterations == 0

    def test_custom_values(self):
        budget = IterationBudget(
            iterations_limit=5,
            tokens_limit=50000,
            time_limit=60.0,
        )
        assert budget.iterations_limit == 5
        assert budget.tokens_limit == 50000
        assert budget.time_limit == 60.0

    def test_reset(self):
        budget = IterationBudget(iterations_limit=10)
        budget.iterations_used = 5
        budget.tokens_prompt = 1000
        budget.tokens_completion = 500
        budget.reset()

        assert budget.iterations_used == 0
        assert budget.tokens_prompt == 0
        assert budget.tokens_completion == 0