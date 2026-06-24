"""run_python 内部工具。"""

from __future__ import annotations

import ast
import subprocess
import sys

from kocor.tools.tool_utils import sanitize_env


class RunPython:
    """在沙盒中执行 Python 代码工具。"""

    NAME = "run_python"
    DESCRIPTION = "在沙盒中执行 Python 代码，只有在没有合适的工具可以完成任务时才使用这个工具。只能使用python标准库。"
    SAFETY_LEVEL = "dangerous"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 代码"},
        },
        "required": ["code"],
    }

    _TIMEOUT = 30

    _STDLIB_MODULES = sys.stdlib_module_names

    @staticmethod
    def _check_stdlib(code: str) -> str | None:
        """检查代码中的 import 是否为 Python 标准库模块。

        Returns:
            如果包含非标准库导入，返回错误消息；否则返回 None。
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "SyntaxError: 代码包含语法错误，无法执行"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split(".")[0]
                    if top_module not in RunPython._STDLIB_MODULES:
                        return (
                            f"ImportError: 只允许使用 Python 标准库模块，"
                            f"不支持第三方库 '{top_module}'。\n"
                            f"请改用标准库中的等效功能。"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_module = node.module.split(".")[0]
                    if top_module not in RunPython._STDLIB_MODULES:
                        return (
                            f"ImportError: 只允许使用 Python 标准库模块，"
                            f"不支持第三方库 '{top_module}'。\n"
                            f"请改用标准库中的等效功能。"
                        )

        return None

    @staticmethod
    def handler(code: str) -> str:
        err = RunPython._check_stdlib(code)
        if err:
            return err

        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                encoding="utf-8",
                timeout=RunPython._TIMEOUT,
                env=sanitize_env(),
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            if result.returncode != 0:
                output = f"Exit code: {result.returncode}\n{output}"
            return output.strip()
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {RunPython._TIMEOUT}s"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"