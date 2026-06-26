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

    provider: str = "openai"                # AI 提供商（openai / anthropic）
    max_iterations: int = 20                # 最大迭代次数
    timeout: int = 300                      # 请求超时秒数
    mcp_config: str = "kocor.mcp.json"      # MCP 服务器配置文件
    skills_config: str = "kocor.skills.json"  # 技能配置文件
    skills_dir: str = "skills"              # 技能目录

    # OpenAI
    openai_api_key: str = ""                # OpenAI API 密钥
    openai_model: str = "gpt-5.5"            # OpenAI 模型名称
    openai_base_url: str = ""               # OpenAI 自定义端点

    # Anthropic
    anthropic_api_key: str = ""             # Anthropic API 密钥
    anthropic_model: str = "opus-4.7"       # Anthropic 模型名称
    anthropic_base_url: str = ""            # Anthropic 自定义端点

    # 上下文管理
    context_strategy: str = "default"       # 上下文策略（default / sliding / summary）
    memory_dir: str = ""                    # 记忆持久化目录（空=不持久化）
    context_max_tokens: int = 200_000       # 上下文最大 token 数
    context_summary_threshold: float = 0.70  # 触发摘要的上下文占用阈值 [0,1]
    context_truncate_threshold: float = 0.90  # 触发截断的上下文占用阈值 [0,1]
    preserve_last_rounds: int = 3           # 保留的最后轮次数量
    preserve_first_rounds: int = 1          # 保留的首轮轮次数量
    token_margin: int = 10_000              # token 余量（预留空间）

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
        """从环境变量加载配置，未设置的字段使用类属性默认值。"""

        provider_raw = os.environ.get("KOCOR_PROVIDER", Config.provider)
        provider = provider_raw.lower()
        valid_providers = {"openai", "anthropic"}
        if provider not in valid_providers:
            raise ValueError(f"不支持的 provider: '{provider}'，可选值: {sorted(valid_providers)}")

        max_iterations_raw = os.environ.get("KOCOR_MAX_ITERATIONS", str(Config.max_iterations))
        try:
            max_iterations = int(max_iterations_raw)
        except ValueError:
            raise ValueError(f"KOCOR_MAX_ITERATIONS 必须是整数，当前值: '{max_iterations_raw}'")
        if max_iterations < 1:
            raise ValueError(f"KOCOR_MAX_ITERATIONS 必须 >= 1，当前值: {max_iterations}")

        timeout_raw = os.environ.get("KOCOR_TIMEOUT", str(Config.timeout))
        try:
            timeout = int(timeout_raw)
        except ValueError:
            raise ValueError(f"KOCOR_TIMEOUT 必须是整数，当前值: '{timeout_raw}'")
        if timeout < 1:
            raise ValueError(f"KOCOR_TIMEOUT 必须 >= 1，当前值: {timeout}")

        mcp_config = os.environ.get("KOCOR_MCP_CONFIG", Config.mcp_config)
        if mcp_config and not os.path.exists(mcp_config):
            raise ValueError(f"KOCOR_MCP_CONFIG 指定的文件不存在: '{mcp_config}'")

        skills_config = os.environ.get("KOCOR_SKILLS_CONFIG", Config.skills_config)
        if skills_config and not os.path.exists(skills_config):
            raise ValueError(f"KOCOR_SKILLS_CONFIG 指定的文件不存在: '{skills_config}'")

        skills_dir = os.environ.get("KOCOR_SKILLS_DIR", Config.skills_dir)

        context_max_tokens_raw = os.environ.get("KOCOR_CONTEXT_MAX_TOKENS", str(Config.context_max_tokens))
        try:
            context_max_tokens = int(context_max_tokens_raw)
        except ValueError:
            raise ValueError(f"KOCOR_CONTEXT_MAX_TOKENS 必须是整数，当前值: '{context_max_tokens_raw}'")
        if context_max_tokens < 1:
            raise ValueError(f"KOCOR_CONTEXT_MAX_TOKENS 必须 >= 1，当前值: {context_max_tokens}")

        preserve_last_rounds_raw = os.environ.get("KOCOR_PRESERVE_LAST_ROUNDS", str(Config.preserve_last_rounds))
        try:
            preserve_last_rounds = int(preserve_last_rounds_raw)
        except ValueError:
            raise ValueError(f"KOCOR_PRESERVE_LAST_ROUNDS 必须是整数，当前值: '{preserve_last_rounds_raw}'")
        if preserve_last_rounds < 0:
            raise ValueError(f"KOCOR_PRESERVE_LAST_ROUNDS 必须 >= 0，当前值: {preserve_last_rounds}")

        preserve_first_rounds_raw = os.environ.get("KOCOR_PRESERVE_FIRST_ROUNDS", str(Config.preserve_first_rounds))
        try:
            preserve_first_rounds = int(preserve_first_rounds_raw)
        except ValueError:
            raise ValueError(f"KOCOR_PRESERVE_FIRST_ROUNDS 必须是整数，当前值: '{preserve_first_rounds_raw}'")
        if preserve_first_rounds < 0:
            raise ValueError(f"KOCOR_PRESERVE_FIRST_ROUNDS 必须 >= 0，当前值: {preserve_first_rounds}")

        token_margin_raw = os.environ.get("KOCOR_TOKEN_MARGIN", str(Config.token_margin))
        try:
            token_margin = int(token_margin_raw)
        except ValueError:
            raise ValueError(f"KOCOR_TOKEN_MARGIN 必须是整数，当前值: '{token_margin_raw}'")
        if token_margin < 0:
            raise ValueError(f"KOCOR_TOKEN_MARGIN 必须 >= 0，当前值: {token_margin}")

        context_summary_threshold_raw = os.environ.get("KOCOR_CONTEXT_SUMMARY_THRESHOLD", str(Config.context_summary_threshold))
        try:
            context_summary_threshold = float(context_summary_threshold_raw)
        except ValueError:
            raise ValueError(f"KOCOR_CONTEXT_SUMMARY_THRESHOLD 必须是数值，当前值: '{context_summary_threshold_raw}'")
        if not 0.0 <= context_summary_threshold <= 1.0:
            raise ValueError(f"KOCOR_CONTEXT_SUMMARY_THRESHOLD 必须在 [0, 1] 范围内，当前值: {context_summary_threshold}")

        context_truncate_threshold_raw = os.environ.get("KOCOR_CONTEXT_TRUNCATE_THRESHOLD", str(Config.context_truncate_threshold))
        try:
            context_truncate_threshold = float(context_truncate_threshold_raw)
        except ValueError:
            raise ValueError(f"KOCOR_CONTEXT_TRUNCATE_THRESHOLD 必须是数值，当前值: '{context_truncate_threshold_raw}'")
        if not 0.0 <= context_truncate_threshold <= 1.0:
            raise ValueError(f"KOCOR_CONTEXT_TRUNCATE_THRESHOLD 必须在 [0, 1] 范围内，当前值: {context_truncate_threshold}")

        return cls(
            provider=provider,
            max_iterations=max_iterations,
            timeout=timeout,
            mcp_config=mcp_config,
            skills_config=skills_config,
            skills_dir=skills_dir,
            openai_api_key=os.environ.get("OPENAI_API_KEY", Config.openai_api_key),
            openai_model=os.environ.get("OPENAI_MODEL", Config.openai_model),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", Config.openai_base_url),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", Config.anthropic_api_key),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", Config.anthropic_model),
            anthropic_base_url=os.environ.get("ANTHROPIC_BASE_URL", Config.anthropic_base_url),
            context_strategy=os.environ.get("KOCOR_CONTEXT_STRATEGY", Config.context_strategy),
            memory_dir=os.environ.get("KOCOR_MEMORY_DIR", Config.memory_dir),
            context_max_tokens=context_max_tokens,
            context_summary_threshold=context_summary_threshold,
            context_truncate_threshold=context_truncate_threshold,
            preserve_last_rounds=preserve_last_rounds,
            preserve_first_rounds=preserve_first_rounds,
            token_margin=token_margin,
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