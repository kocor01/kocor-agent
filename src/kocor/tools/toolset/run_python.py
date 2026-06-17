"""run_python 内部工具。"""

from __future__ import annotations

import subprocess

from kocor.tools.tool_utils import sanitize_env

NAME = "run_python"
DESCRIPTION = "在沙盒中执行 Python 代码"
PARAMETERS = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python 代码"},
    },
    "required": ["code"],
}

_TIMEOUT = 30


def handler(code: str) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=sanitize_env(),
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output = f"Exit code: {result.returncode}\n{output}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def toolRegistry_to(toolRegistry) -> None:
    toolRegistry.register(
        name=NAME,
        description=DESCRIPTION,
        parameters=PARAMETERS,
        handler=handler,
    )
