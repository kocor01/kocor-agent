"""子代理摘要提取与字符上限裁剪。"""


def truncate_summary(text: str, max_chars: int) -> str:
    """截断摘要到字符上限，保留头 75% + 尾 25%（按行边界对齐）。

    Args:
        text: 原始摘要文本
        max_chars: 字符上限（0 = 禁用截断）

    Returns:
        截断后文本（含截断标记），或原始文本。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    head_chars = int(max_chars * 0.75)
    tail_chars = max_chars - head_chars

    # 取前 head_chars 字符，按行边界 snap 到最近行尾
    head = text[:head_chars]
    head_lines = head.splitlines()
    # 丢弃最后一行（可能不完整），保证完整行
    if len(head_lines) > 1:
        head = "\n".join(head_lines[:-1])
    elif len(head) > 0:
        # 只有一行，截断到 head_chars
        head = head[:head_chars]

    # 取尾 tail_chars 字符，按行边界 snap 到最近行首
    tail = text[-tail_chars:]
    tail_lines = tail.splitlines()
    if len(tail_lines) > 1:
        # 丢弃第一行（可能不完整）
        tail = "\n".join(tail_lines[1:])
    elif len(tail) > 0:
        tail = tail[-tail_chars:]

    truncated = f"{head}\n[... {len(text) - max_chars} characters truncated ...]\n{tail}"
    return truncated


def extract_summary(
    final_text: str | None,
    max_chars: int = 8000,
    status: str = "completed",
) -> dict:
    """从子代理最终回复中提取摘要，构建结构化结果。

    Args:
        final_text: 子代理最后一条 assistant 消息的文本内容
        max_chars: 摘要字符上限
        status: 子代理完成状态（completed/budget_exhausted/interrupted/error）

    Returns:
        结构化摘要字典（含 status、summary）
    """
    summary = final_text or ""
    if max_chars > 0 and len(summary) > max_chars:
        summary = truncate_summary(summary, max_chars)
    return {"status": status, "summary": summary}