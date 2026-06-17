"""MCP 工具输出截断。

三级截断策略：
1. 单行截断 — 超长行末尾截断
2. 行数截断 — 超出保留行数的部分头尾各保留 50%
3. 字节截断 — 超出最大字节数的部分头尾各保留 50%
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TruncateConfig:
    max_bytes: int = 50_000
    max_lines: int = 2_000
    max_line_length: int = 2_000


def truncate_output(text: str, config: TruncateConfig | None = None) -> str:
    """对工具输出进行三级截断。

    Args:
        text: 原始输出
        config: 截断配置，None 则使用默认值

    Returns:
        截断后的文本
    """
    if not text:
        return text

    cfg = config or TruncateConfig()

    # 阶段 1：单行截断
    lines = text.split("\n")
    truncated_lines = []
    for line in lines:
        if len(line) > cfg.max_line_length:
            line = line[:cfg.max_line_length] + "... [truncated]"
        truncated_lines.append(line)

    # 阶段 2：行数截断
    if len(truncated_lines) > cfg.max_lines:
        head_count = int(cfg.max_lines * 0.5)
        tail_count = cfg.max_lines - head_count
        head = truncated_lines[:head_count]
        tail = truncated_lines[-tail_count:]
        omitted = len(truncated_lines) - cfg.max_lines
        truncated_lines = head + [
            f"\n[... {omitted} lines truncated ...]\n"
        ] + tail

    result = "\n".join(truncated_lines)

    # 阶段 3：字节截断
    if len(result.encode("utf-8")) > cfg.max_bytes:
        half = cfg.max_bytes // 2
        head = result[:half]
        tail = result[-half:]
        result = f"{head}\n[... OUTPUT TRUNCATED ...]\n{tail}"

    return result
