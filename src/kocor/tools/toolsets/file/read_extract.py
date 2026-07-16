"""文档提取模块。

纯标准库实现，零外部依赖。
支持 Jupyter Notebook (.ipynb) 的单元格文本提取。

用于 read_file 工具的可选文本提取路径。
"""

from __future__ import annotations

import json
import os
from typing import Any

EXTRACTABLE_EXTENSIONS: frozenset[str] = frozenset({".ipynb"})


class ExtractionError(Exception):
    """文档提取失败时抛出。"""


def is_extractable_document(path: str) -> bool:
    """检查文件是否为可提取文档。"""
    ext = os.path.splitext(path)[1].lower()
    return ext in EXTRACTABLE_EXTENSIONS


def extract_document_text(path: str) -> str:
    """从文档中提取文本内容。

    支持格式：
    - .ipynb: Jupyter Notebook → 单元格分割输出

    Args:
        path: 文件路径

    Returns:
        提取的文本内容

    Raises:
        ExtractionError: 提取失败
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ipynb":
        return _extract_notebook(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def _extract_notebook(path: str) -> str:
    """提取 Jupyter Notebook 的单元格内容。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise ExtractionError(f"Not a valid notebook: {e}") from e

    if not isinstance(nb, dict):
        raise ExtractionError("Notebook root is not an object")

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ExtractionError("Notebook cells is not a list")

    parts: list[str] = []
    for i, cell in enumerate(cells, 1):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])

        # source 可以是字符串或字符串列表
        source_text = _source_text(source)
        if not source_text.strip():
            continue

        # 单元格头
        parts.append(f"[{cell_type}] cell {i}:")
        parts.append(source_text)

        # 提取输出（仅 code cell）
        if cell_type == "code" and "outputs" in cell:
            outputs = cell.get("outputs", [])
            output_text = _extract_outputs(outputs)
            if output_text:
                parts.append("[output]")
                parts.append(output_text)

        parts.append("")  # 空行分隔

    return "\n".join(parts).rstrip()


def _source_text(source: Any) -> str:
    """将 source 字段转为字符串。"""
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(item) for item in source if isinstance(item, str))
    return ""


def _extract_outputs(outputs: list[dict]) -> str:
    """提取单元格输出。"""
    text_parts: list[str] = []
    for output in outputs:
        output_type = output.get("output_type", "")

        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if text:
                text_parts.append(text)

        elif output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            # 优先提取 text/plain
            text = data.get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
            if text:
                text_parts.append(text)

        elif output_type == "error":
            ename = output.get("ename", "")
            evalue = output.get("evalue", "")
            traceback_list = output.get("traceback", [])
            if traceback_list:
                text_parts.append("\n".join(traceback_list))
            else:
                text_parts.append(f"{ename}: {evalue}")

    return "\n".join(text_parts).rstrip()