"""配置加载。

从环境变量读取配置，提供默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """系统配置。"""

    provider: str = "openai"
    max_iterations: int = 20
    timeout: int = 30
    mcp_config: str = "kocor.mcp.json"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = ""

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_base_url: str = ""



def load_config() -> Config:
    """从环境变量加载配置，未设置的字段使用 Config 字段默认值。

    环境变量:
        KOCOR_PROVIDER: provider 选择（支持 openai / OpenAI / anthropic / Anthropic）
        KOCOR_MAX_ITERATIONS: 最大迭代次数
        KOCOR_TIMEOUT: 超时秒数
        KOCOR_MCP_CONFIG: MCP 服务器配置文件路径

    Returns:
        配置对象

    Raises:
        ValueError: provider 非法、整数参数无效，或 MCP 配置文件不存在
    """
    provider_raw = os.environ.get("KOCOR_PROVIDER")
    if provider_raw is not None:
        provider = provider_raw.lower()
        valid_providers = {"openai", "anthropic"}
        if provider not in valid_providers:
            raise ValueError(f"不支持的 provider: '{provider}'，可选值: {sorted(valid_providers)}")
    else:
        provider = "openai"

    max_iterations_raw = os.environ.get("KOCOR_MAX_ITERATIONS")
    if max_iterations_raw is not None:
        try:
            max_iterations = int(max_iterations_raw)
        except ValueError:
            raise ValueError(f"KOCOR_MAX_ITERATIONS 必须是整数，当前值: '{max_iterations_raw}'")
        if max_iterations < 1:
            raise ValueError(f"KOCOR_MAX_ITERATIONS 必须 >= 1，当前值: {max_iterations}")
    else:
        max_iterations = 20

    timeout_raw = os.environ.get("KOCOR_TIMEOUT")
    if timeout_raw is not None:
        try:
            timeout = int(timeout_raw)
        except ValueError:
            raise ValueError(f"KOCOR_TIMEOUT 必须是整数，当前值: '{timeout_raw}'")
        if timeout < 1:
            raise ValueError(f"KOCOR_TIMEOUT 必须 >= 1，当前值: {timeout}")
    else:
        timeout = 30

    mcp_config_raw = os.environ.get("KOCOR_MCP_CONFIG")
    if mcp_config_raw is not None:
        if mcp_config_raw and not os.path.exists(mcp_config_raw):
            raise ValueError(f"KOCOR_MCP_CONFIG 指定的文件不存在: '{mcp_config_raw}'")
        mcp_config = mcp_config_raw
    else:
        mcp_config = "kocor.mcp.json"

    return Config(
        provider=provider,
        max_iterations=max_iterations,
        timeout=timeout,
        mcp_config=mcp_config,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        anthropic_base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
    )