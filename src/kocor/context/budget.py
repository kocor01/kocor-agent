"""Token 预算与使用统计。

独立文件，供上下文策略和构建器使用。
"""

from __future__ import annotations

from dataclasses import dataclass

from kocor.config import config_get


@dataclass
class TokenBudget:
    """Token 预算与使用统计。

    Attributes:
        limit: 上下文窗口上限 token 数
        used_prompt: 当前 prompt 已用 token
        used_completion: 当前 completion 已用 token
        threshold_summary: 触发摘要的阈值比例（0.0 ~ 1.0）
        threshold_truncate: 触发截断的阈值比例（0.0 ~ 1.0）
    """

    limit: int = config_get("context_max_tokens")
    used_prompt: int = 0
    used_completion: int = 0
    threshold_summary: float = config_get("context_summary_threshold")
    threshold_truncate: float = config_get("context_truncate_threshold")

    @property
    def remaining(self) -> int:
        return self.limit - self.used_prompt

    @property
    def usage_ratio(self) -> float:
        if self.limit <= 0:
            return 0.0
        return self.used_prompt / self.limit

    def should_summarize(self) -> bool:
        return self.usage_ratio >= self.threshold_summary

    def should_truncate(self) -> bool:
        return self.usage_ratio >= self.threshold_truncate