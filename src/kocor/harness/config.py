"""Harness 配置。"""

from dataclasses import dataclass


@dataclass
class HarnessConfig:
    """Harness 运行时配置。

    控制 Agent 循环、权限、上下文管理、沙箱、
    可观测性和重试行为。从 JSON 配置文件、
    环境变量和 CLI 参数加载（优先级递增）。
    """

    # 循环控制
    max_iterations: int = 20
    max_tokens_per_response: int = 4096
    max_total_time: int = 300

    # 权限
    permission_policy: str = "default"  # permissive | default | strict
                                        # permissive: safe/caution 自动允许，dangerous 询问一次
                                        # default: safe 自动允许，caution/dangerous 询问一次
                                        # strict: 全部检查，dangerous 默认拒绝
    permission_cache: bool = True

    # 上下文
    context_max_tokens: int = 200_000
    context_summary_threshold: float = 0.70
    context_truncate_threshold: float = 0.90
    preserve_rounds: int = 3

    # 沙箱
    sandbox_timeout: int = 30
    sandbox_memory_limit: str = "256m"
    sandbox_blocked_modules: list[str] | None = None
    sandbox_network: bool = False

    # 工具
    allowed_dir: str = ""

    # 重试
    max_retries: int = 3
    retry_delay_base: float = 1.0