"""read_file 内部工具。"""

from __future__ import annotations

import os

from kocor.tools.tool_utils import resolve_safe_path


class ReadFile:
    """读取文件内容工具。"""

    NAME = "read_file"
    DESCRIPTION = "读取文件内容"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
        },
        "required": ["path"],
    }

    @staticmethod
    def handler(path: str) -> str:
        allowed_dir = os.path.realpath(os.getcwd())
        safe_path = resolve_safe_path(path, allowed_dir)
        if not os.path.exists(safe_path):
            return f"Error: file not found: {path}"
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()