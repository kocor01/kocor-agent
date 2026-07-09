"""命令安全审查模块：危险命令检测、workdir 验证。"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# 硬阻断模式：无论如何都不允许执行
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # 文件系统破坏
    (r"\brm\s+-rf\s+(?:/\s*$|/$)", "Dangerous: recursive root deletion"),
    (r"\bmkfs\b", "Dangerous: filesystem formatting"),
    (r"\bdd\s+if=/dev/zero\b", "Dangerous: disk overwrite"),
    # 权限提升/系统篡改
    (r"\busermod\b", "Dangerous: user modification"),
    (r"\bpasswd\b", "Dangerous: password change"),
    # 加密/挖矿
    (r"\bxmrig\b", "Dangerous: cryptocurrency miner"),
    (r"\bminerd\b", "Dangerous: cryptocurrency miner"),
    (r"\bcryptominer\b", "Dangerous: cryptocurrency miner"),
    # 数据泄露
    (r"\bcurl\s+.*\|\s*(?:bash|sh)\b", "Dangerous: pipe curl to shell"),
    (r"\bwget\s+.*\|\s*(?:bash|sh)\b", "Dangerous: pipe wget to shell"),
]

# ---------------------------------------------------------------------------
# 需要审批模式：default/strict 策略下需要交互式批准
# ---------------------------------------------------------------------------

REQUIRE_APPROVAL_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-rf\b", "Approval needed: recursive delete"),
    (r"\brm\s+--recursive\b", "Approval needed: recursive delete"),
    (r"\bchmod\s+-R\b", "Approval needed: recursive chmod"),
    (r"\bchown\s+-R\b", "Approval needed: recursive chown"),
    (r"\bkill\b", "Approval needed: kill process"),
    (r"\bkillall\b", "Approval needed: killall process"),
    (r"\bpkill\b", "Approval needed: pkill process"),
    (r"\bnohup\b", "Approval needed: nohup background"),
    (r"\bwget\b", "Approval needed: wget download"),
    (r"\bchmod\s+4[0-7][0-7]\s", "Approval needed: file permission change"),
    (r"\bsudo\s+(?!-S\s)", "Approval needed: sudo (use -S flag)"),
]

# ---------------------------------------------------------------------------
# workdir 字符白名单：仅允许安全的文件系统路径字符
# ---------------------------------------------------------------------------

_WORKDIR_SAFE_RE = re.compile(r'^[A-Za-z0-9/\\:_\-.~ +@=,]+$')


def detect_dangerous_command(command: str) -> tuple[str, str]:
    """检测命令风险等级。

    Returns:
        ("safe"/"caution"/"dangerous", 原因描述或 "")
    """
    if not command:
        return "safe", ""

    # 先检查硬阻断模式
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return "dangerous", reason

    # 再检查需要审批的操作
    for pattern, reason in REQUIRE_APPROVAL_PATTERNS:
        if re.search(pattern, command):
            return "caution", reason

    return "safe", ""


def validate_workdir(workdir: Optional[str]) -> Optional[str]:
    """验证 workdir 安全性（字符白名单）。

    Returns:
        None 表示安全，字符串为错误消息。
    """
    if not workdir:
        return None
    if not _WORKDIR_SAFE_RE.match(workdir):
        for ch in workdir:
            if not _WORKDIR_SAFE_RE.match(ch):
                return (
                    f"Blocked: workdir contains disallowed character {repr(ch)}. "
                    "Use a simple filesystem path without shell metacharacters."
                )
        return "Blocked: workdir contains disallowed characters."
    return None