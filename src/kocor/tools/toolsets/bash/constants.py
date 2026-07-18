"""Shell 查找、路径常量、子进程环境构建。"""

import os
import platform
import re
import shlex
import shutil
import tempfile

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# 平台检测
# ---------------------------------------------------------------------------


def _is_windows() -> bool:
    """判断当前系统是否为 Windows。"""
    return _IS_WINDOWS


IS_WINDOWS = _is_windows()

# ---------------------------------------------------------------------------
# Shell 查找
# ---------------------------------------------------------------------------


def _find_bash() -> str:
    """跨平台查找 bash 二进制。

    - POSIX: shutil.which("bash") → /bin/bash（兜底）
    - Windows: KOCOR_GIT_BASH_PATH -> Program Files/Git/bin/bash.exe -> shutil.which
    """
    if not IS_WINDOWS:
        return shutil.which("bash") or "/bin/bash"

    # Windows: 优先使用环境变量指定的 Git Bash 路径
    custom = os.environ.get("KOCOR_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        return custom

    # 查找常见 Git for Windows 安装路径
    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    found = shutil.which("bash")
    if found:
        return found

    raise RuntimeError(
        "bash not found. Install Git for Windows: https://git-scm.com/download/win"
    )


# ---------------------------------------------------------------------------
# 子进程环境构建
# ---------------------------------------------------------------------------

# 已知的敏感变量后缀/前缀
_SENSITIVE_KEY_SUFFIXES = ("_API_KEY", "_SECRET", "_TOKEN")
_SENSITIVE_KEY_PREFIXES = ("OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "GEMINI_", "COHERE_")


def _is_sensitive_key(key: str) -> bool:
    """判断环境变量名是否可能是敏感信息。"""
    upper = key.upper()
    for suffix in _SENSITIVE_KEY_SUFFIXES:
        if upper.endswith(suffix):
            return True
    for prefix in _SENSITIVE_KEY_PREFIXES:
        if upper.startswith(prefix) and upper.endswith(("_API_KEY", "_BASE_URL", "_MODEL")):
            return True
    return False


def _make_run_env() -> dict[str, str]:
    """构建子进程环境：继承主机环境，过滤敏感变量。

    移除子进程中不需要的 API Key、Token 等敏感变量，
    确保 PYTHONUNBUFFERED=1 以便实时读取输出。
    """
    env = os.environ.copy()
    # 移除敏感变量
    for key in list(env):
        if _is_sensitive_key(key):
            env.pop(key, None)
    # 确保 PYTHONUNBUFFERED 以支持实时输出
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ---------------------------------------------------------------------------
# 安全 CWD 解析
# ---------------------------------------------------------------------------


def _resolve_safe_cwd(cwd: str) -> str:
    """返回 *cwd* 如果存在且是目录；否则向上查找最近的存在的祖先目录。

    当 CWD 被删除时（如之前调用的 rm -rf 删除工作目录），
    Popen 的 cwd 参数会抛出 FileNotFoundError。
    此函数从给定 CWD 向上遍历直到找到存在的目录。

    Windows 上同时处理 MSYS 风格路径（/c/Users/...）到 Windows 原生格式的转换。
    """
    if not cwd:
        return tempfile.gettempdir()

    # Windows：尝试将 MSYS 路径（/c/Users/...）转换为 Windows 原生格式（C:\Users\...）
    # MSYS 路径在 Git Bash 内部使用，但 subprocess.Popen 的 cwd 需要 Windows 原生路径。
    resolved = cwd
    if IS_WINDOWS:
        m = re.match(r'^/([a-zA-Z])/(.*)', cwd)
        if m:
            resolved = f"{m.group(1).upper()}:\\{m.group(2).replace('/', '\\')}"

    resolved = os.path.realpath(resolved)
    if os.path.isdir(resolved):
        return resolved

    parent = os.path.dirname(resolved)
    while parent:
        if os.path.isdir(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        parent = next_parent
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# cd 路径引号辅助
# ---------------------------------------------------------------------------


def _quote_cwd_for_cd(cwd: str) -> str:
    """为 `cd` 命令引号处理 CWD 路径，同时保留 `~` 展开。

    - ``~`` → ``~``（原样）
    - ``~/`` → ``$HOME``
    - ``~/foo bar`` → ``$HOME/foo bar``（引号保留）
    - ``/abs/path`` → ``shlex.quote()``
    """
    if cwd == "~":
        return cwd
    if cwd == "~/":
        return "$HOME"
    if cwd.startswith("~/"):
        return f"$HOME/{shlex.quote(cwd[2:])}"
    return shlex.quote(cwd)