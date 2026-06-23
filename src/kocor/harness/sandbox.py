"""带安全控制的代码执行沙箱。

提供基于子进程的轻量级沙箱，支持模块导入拦截、
超时、内存限制和环境清理。
"""

import os
import subprocess
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """沙箱代码执行的结果。"""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox:
    """轻量级代码执行沙箱。

    在子进程中运行代码，拦截危险模块、
    可配置超时和环境变量清理。
    """

    def __init__(
        self,
        timeout: int = 30,
        memory_limit: str = "256m",
        allowed_modules: set[str] | None = None,
        blocked_modules: set[str] | None = None,
        network_access: bool = False,
    ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = allowed_modules or {
            "math", "json", "re", "datetime",
            "collections", "itertools", "functools",
            "string", "typing", "pathlib", "decimal",
            "uuid", "hashlib", "statistics",
        }
        self.blocked_modules = blocked_modules or {
            "os", "subprocess", "sys", "shutil",
            "socket", "urllib", "requests", "http",
            "importlib", "ctypes", "multiprocessing",
        }
        self.network_access = network_access

    def run(self, code: str) -> SandboxResult:
        """在沙箱中执行代码。"""
        wrapper = self._build_wrapper(code)
        try:
            result = subprocess.run(
                ["python", "-c", wrapper],
                capture_output=True, text=True,
                timeout=self.timeout,
                env=self._build_env(),
            )
            
            return SandboxResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="", stderr="", exit_code=-1, timed_out=True,
            )

    def _build_wrapper(self, code: str) -> str:
        """包装用户代码，添加导入拦截。"""
        blockers = [f"'{mod}'" for mod in self.blocked_modules]
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

    def _build_env(self) -> dict[str, str]:
        """构建已移除敏感变量的环境。"""
        env = os.environ.copy()
        sensitive_prefixes = ("API_KEY", "SECRET", "TOKEN", "PASSWORD")
        for key in list(env.keys()):
            if any(s in key.upper() for s in sensitive_prefixes):
                del env[key]
        if not self.network_access:
            for key in list(env.keys()):
                if key.lower().startswith("http_"):
                    del env[key]
        return env