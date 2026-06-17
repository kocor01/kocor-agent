"""write_file 内部工具。"""

from __future__ import annotations

import os

from kocor.tools.tool_utils import resolve_safe_path

NAME = "write_file"
DESCRIPTION = "写入文件内容"
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "文件路径"},
        "content": {"type": "string", "description": "文件内容"},
    },
    "required": ["path", "content"],
}


def handler(path: str, content: str) -> str:
    allowed_dir = os.path.realpath(os.getcwd())
    safe_path = resolve_safe_path(path, allowed_dir)
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    with open(safe_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Success: wrote {len(content)} bytes to {path}"


def register_to(registry) -> None:
    registry.register(
        name=NAME,
        description=DESCRIPTION,
        parameters=PARAMETERS,
        handler=handler,
    )
