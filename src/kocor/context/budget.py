"""Token 预算与使用统计。

独立文件，供上下文策略和构建器使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kocor.config import Config


@dataclass
class TokenBudget:
    """Token 预算与使用统计。

    Attributes:
        limit: 上下文窗口上限 token 数
        used_prompt: 当前 prompt 已用 token
        threshold_summary: 触发摘要的阈值比例（0.0 ~ 1.0）
        threshold_truncate: 触发截断的阈值比例（0.0 ~ 1.0）
    """

    limit: int = field(default_factory=lambda: Config.load().context_max_tokens)
    used_prompt: int = 0
    threshold_summary: float = field(default_factory=lambda: Config.load().context_summary_threshold)
    threshold_truncate: float = field(default_factory=lambda: Config.load().context_truncate_threshold)

    @property
    def remaining(self) -> int:
        """剩余可用 token 数。"""
        return self.limit - self.used_prompt

    @property
    def usage_ratio(self) -> float:
        """当前使用比例（0.0 ~ 1.0）。"""
        if self.limit <= 0:
            return 0.0
        return self.used_prompt / self.limit

    def should_summarize(self) -> bool:
        """是否触发摘要（使用率 ≥ 摘要阈值）。"""
        return self.usage_ratio >= self.threshold_summary

    def should_truncate(self) -> bool:
        """是否触发截断（使用率 ≥ 截断阈值）。"""
        return self.usage_ratio >= self.threshold_truncate

    def reset(self) -> None:
        """重置已用 token 计数。"""
        self.used_prompt = 0