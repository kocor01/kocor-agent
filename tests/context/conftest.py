"""上下文测试模块的共享配置。"""

import pytest

from kocor.harness.logger import setup_logger


@pytest.fixture(autouse=True, scope="session")
def init_logger():
    setup_logger(level="CRITICAL")
