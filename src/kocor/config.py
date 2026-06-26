"""配置加载。

从环境变量读取配置，提供默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, Optional


@dataclass
class Config:
    """系统配置。"""

    _instance: ClassVar[Optional[Config]] = None

    provider: str = "openai"
    max_iterations: int = 20
    timeout: int = 30
    mcp_config: str = "kocor.mcp.json"
    skills_config: str = "kocor.skills.json"
    skills_dir: str = "skills"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = ""

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_base_url: str = ""

    # 上下文管理
    context_strategy: str = "default"
    memory_dir: str = ""
    project_instructions_path: str = "KOCOR.md"
    context_max_tokens: int = 200_000
    context_summary_threshold: float = 0.70
    context_truncate_threshold: float = 0.90
    preserve_rounds: int = 3

    @classmethod
    def load(cls) -> Config:
        """获取全局配置，首次调用时从环境变量加载。"""
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """清除全局实例（用于测试）。"""
        cls._instance = None

    @classmethod
    def _load(cls) -> Config:
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

        skills_config_raw = os.environ.get("KOCOR_SKILLS_CONFIG")
        if skills_config_raw is not None:
            skills_config = skills_config_raw
        else:
            skills_config = "kocor.skills.json"

        skills_dir_raw = os.environ.get("KOCOR_SKILLS_DIR")
        if skills_dir_raw is not None:
            skills_dir = skills_dir_raw
        else:
            skills_dir = "skills"

        context_max_tokens_raw = os.environ.get("KOCOR_CONTEXT_MAX_TOKENS")
        if context_max_tokens_raw is not None:
            try:
                context_max_tokens = int(context_max_tokens_raw)
            except ValueError:
                context_max_tokens = 200_000
        else:
            context_max_tokens = 200_000

        preserve_rounds_raw = os.environ.get("KOCOR_PRESERVE_ROUNDS")
        if preserve_rounds_raw is not None:
            try:
                preserve_rounds = int(preserve_rounds_raw)
            except ValueError:
                preserve_rounds = 3
        else:
            preserve_rounds = 3

        context_summary_threshold_raw = os.environ.get("KOCOR_CONTEXT_SUMMARY_THRESHOLD")
        if context_summary_threshold_raw is not None:
            try:
                context_summary_threshold = float(context_summary_threshold_raw)
            except ValueError:
                context_summary_threshold = 0.70
        else:
            context_summary_threshold = 0.70

        context_truncate_threshold_raw = os.environ.get("KOCOR_CONTEXT_TRUNCATE_THRESHOLD")
        if context_truncate_threshold_raw is not None:
            try:
                context_truncate_threshold = float(context_truncate_threshold_raw)
            except ValueError:
                context_truncate_threshold = 0.90
        else:
            context_truncate_threshold = 0.90

        return cls(
            provider=provider,
            max_iterations=max_iterations,
            timeout=timeout,
            mcp_config=mcp_config,
            skills_config=skills_config,
            skills_dir=skills_dir,
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            anthropic_base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
            context_strategy=os.environ.get("KOCOR_CONTEXT_STRATEGY", "default"),
            memory_dir=os.environ.get("KOCOR_MEMORY_DIR", ""),
            project_instructions_path=os.environ.get("KOCOR_PROJECT_INSTRUCTIONS_PATH", "KOCOR.md"),
            context_max_tokens=context_max_tokens,
            context_summary_threshold=context_summary_threshold,
            context_truncate_threshold=context_truncate_threshold,
            preserve_rounds=preserve_rounds,
        )


def config_get(key: str, default: Any = None) -> Any:
    """快速获取单个配置项的值。

    Args:
        key: 配置项名称（如 "provider", "max_iterations"）
        default: 配置项不存在时返回的默认值

    Returns:
        配置项的值

    Examples:
        >>> get_config("provider")
        'openai'
        >>> get_config("max_iterations")
        20
    """
    return getattr(Config.load(), key, default)