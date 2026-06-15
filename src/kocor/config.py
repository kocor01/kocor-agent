"""配置加载。

从环境变量读取配置，提供默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 客户端配置。

    只保留 provider 选择，model 和 base_url 下放到各 client 实例化时读取。
    """

    provider: str = "openai"
    max_iterations: int = 20
    timeout: int = 30


def load_config() -> LLMConfig:
    """从环境变量加载配置。

    环境变量:
        KOCOR_PROVIDER: provider 选择（支持 openai / OpenAI / anthropic / Anthropic）
        KOCOR_MAX_ITERATIONS: 最大迭代次数
        KOCOR_TIMEOUT: 超时秒数

    Returns:
        配置对象

    Raises:
        ValueError: provider 非法或整数参数无效
    """
    raw = os.environ.get("KOCOR_PROVIDER", "openai").lower()
    valid_providers = {"openai", "anthropic"}
    if raw not in valid_providers:
        raise ValueError(f"不支持的 provider: '{raw}'，可选值: {sorted(valid_providers)}")

    max_iterations_raw = os.environ.get("KOCOR_MAX_ITERATIONS", "20")
    try:
        max_iterations = int(max_iterations_raw)
    except ValueError:
        raise ValueError(f"KOCOR_MAX_ITERATIONS 必须是整数，当前值: '{max_iterations_raw}'")
    if max_iterations < 1:
        raise ValueError(f"KOCOR_MAX_ITERATIONS 必须 >= 1，当前值: {max_iterations}")

    timeout_raw = os.environ.get("KOCOR_TIMEOUT", "30")
    try:
        timeout = int(timeout_raw)
    except ValueError:
        raise ValueError(f"KOCOR_TIMEOUT 必须是整数，当前值: '{timeout_raw}'")
    if timeout < 1:
        raise ValueError(f"KOCOR_TIMEOUT 必须 >= 1，当前值: {timeout}")

    return LLMConfig(
        provider=raw,
        max_iterations=max_iterations,
        timeout=timeout,
    )
