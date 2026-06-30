"""run_python 内部工具。

在子进程中执行 Python 代码，提供双层安全保护：
1. AST 静态检查：只允许标准库模块导入
2. 运行时拦截：通过 __import__ 包装器阻断危险模块
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

from kocor.tools.permission import PermissionManager


class RunPython:
    """在沙盒中执行 Python 代码工具。"""

    NAME = "run_python"
    DESCRIPTION = "在沙盒中执行 Python 代码，只有在没有合适的工具可以完成任务时才使用这个工具。只能使用python标准库。"
    SAFETY_LEVEL = PermissionManager.SAFETY_DANGEROUS
    PARAMETERS = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 代码"},
        },
        "required": ["code"],
    }

    _TIMEOUT = 30

    _STDLIB_MODULES = sys.stdlib_module_names

    _BLOCKED_MODULES: frozenset[str] = frozenset({
        "os", "subprocess", "sys", "shutil",
        "socket", "urllib", "http",
        "importlib", "ctypes", "multiprocessing",
    })

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
    def _build_wrapper(code: str) -> str:
        """包装用户代码，添加运行时模块导入拦截。"""
        blockers = [f"'{mod}'" for mod in sorted(RunPython._BLOCKED_MODULES)]
        return f"""\
import builtins as __builtins__
_original_import = __builtins__.__import__

def _safe_import(name, *args, **kwargs):
    blocked = {{{','.join(blockers)}}}
    if name.split('.')[0] in blocked:
        raise ImportError(f"Module '{{name}}' is blocked for security reasons")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = _safe_import

{code}
"""

    @staticmethod
    def _build_env() -> dict[str, str]:
        """创建不含敏感凭证的环境变量副本。"""
        env = os.environ.copy()
        _sensitive_keys = [
            key
            for key in env
            if key.endswith(("_API_KEY", "_SECRET", "_TOKEN"))
            or key in ("OPENAI_ORG_ID",)
            or any(s in key.upper() for s in ("API_KEY", "SECRET", "TOKEN", "PASSWORD"))
        ]
        for key in _sensitive_keys:
            env.pop(key, None)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def handler(code: str) -> str:
        err = RunPython._check_stdlib(code)
        if err:
            return err

        wrapper = RunPython._build_wrapper(code)

        try:
            result = subprocess.run(
                ["python", "-c", wrapper],
                capture_output=True,
                encoding="utf-8",
                timeout=RunPython._TIMEOUT,
                env=RunPython._build_env(),
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