"""命令安全审查模块：危险命令检测、workdir 验证。

提供三层防护：
1. 硬阻断模式 — 无论如何都不允许执行
2. 需要审批模式 — 默认/strict 策略下需要用户确认
3. 命令规范化 — 在检测前处理常见的 shell 混淆绕过（引号分段、base64 等）

规范化只用于模式匹配检测，实际执行的命令不变。
"""

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
    # 数据泄露：管道到 shell
    (r"\bcurl\s+.*\|\s*(?:bash|sh)\b", "Dangerous: pipe curl to shell"),
    (r"\bwget\s+.*\|\s*(?:bash|sh)\b", "Dangerous: pipe wget to shell"),
    # Base64 解码后管道到 shell
    (r"\bbase64\s+(?:-d|--decode)\s*\|\s*(?:bash|sh)\b", "Dangerous: base64 decode to shell"),
    (r"\|\s*base64\s+(?:-d|--decode)\s*\|\s*(?:bash|sh)\b", "Dangerous: piped base64 decode to shell"),
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


# ---------------------------------------------------------------------------
# 命令规范化：移除 shell 混淆以便模式匹配
# ---------------------------------------------------------------------------

# 编译一次提高性能
_RE_ADJACENT_SQ = re.compile(r"'([^']*)''([^']*)'")     # 'a''b' → ab
_RE_ADJACENT_DQ = re.compile(r'"([^"]*)""([^"]*)"')     # "a""b" → ab
_RE_ADJACENT_MIXED_1 = re.compile(r"'([^']*)'\"([^\"]*)\"")  # 'a'"b" → ab
_RE_ADJACENT_MIXED_2 = re.compile(r'"([^"]*)"\'([^\']*)\'')  # "a"'b' → ab


def _normalize_command(command: str) -> str:
    """规范化命令以便危险模式检测：移除 shell 混淆（引号片段拼接等）。

    该规范化仅用于**模式匹配**，实际执行的命令保持不变。
    处理以下绕过方式：
    - 空引号移除：r''m → rm
    - 相邻引号片段合并：'r''m' → rm、"r""m" → rm、'r'"m" → rm
    - 单词内单引号剥离：r'm' → rm（但不剥离含空格的引用参数）
    """
    if not command:
        return command

    # 1. 折叠空白（多空格/tab → 单空格）
    cmd = re.sub(r'\s+', ' ', command).strip()

    # 2. 移除空引号
    cmd = cmd.replace("''", "").replace('""', '')

    # 3. 循环合并相邻引号片段（最多 10 轮防死循环）
    for _ in range(10):
        before = cmd
        cmd = _RE_ADJACENT_SQ.sub(lambda m: m.group(1) + m.group(2), cmd)
        cmd = _RE_ADJACENT_DQ.sub(lambda m: m.group(1) + m.group(2), cmd)
        cmd = _RE_ADJACENT_MIXED_1.sub(lambda m: m.group(1) + m.group(2), cmd)
        cmd = _RE_ADJACENT_MIXED_2.sub(lambda m: m.group(1) + m.group(2), cmd)
        if cmd == before:
            break

    # 4. 移除单词间的单引号（shell 中 r'm' → rm，只是字符保护）
    cmd = re.sub(r"(\w)'(\w)", r'\1\2', cmd)

    # 5. 剥离不含空格的单引号内容：'rm' → rm（'rm -rf /' 含空格 → 保留）
    cmd = re.sub(r"'([^\s']+)'", r'\1', cmd)

    # 6. 剥离不含空格的双引号内容："rm" → rm（仅用于模式匹配检测）
    cmd = re.sub(r'"([^\s"]+)"', r'\1', cmd)

    # 7. 处理剥离后遗留的单引号残片：'X'Y'Z' → XYZ（循环清理不匹配的边界引号）
    #    '-'rf' 剥离 '-' → -rf' → 移除尾随引号 → -rf
    cmd = re.sub(r"'(\s|$)", r'\1', cmd)   # 词尾不匹配引号：rf' → rf
    cmd = re.sub(r"(\s)'", r'\1', cmd)     # 词首不匹配引号：'r → r

    # 8. 再次折叠空白（合并后可能有新多余空白）
    cmd = re.sub(r'\s+', ' ', cmd).strip()
    return cmd


def detect_dangerous_command(command: str) -> tuple[str, str]:
    """检测命令风险等级。

    先检查原始命令，再检查规范化后的命令（防绕过）。

    Returns:
        ("safe"/"caution"/"dangerous", 原因描述或 "")
    """
    if not command:
        return "safe", ""

    # 规范化（仅用于模式匹配检测，不影响实际执行）
    normalized = _normalize_command(command)

    # 检查原始命令和规范化版本的并集（优先用原始命令检测 "dangerous"）
    for cmd in (command, normalized):
        for pattern, reason in DANGEROUS_PATTERNS:
            if re.search(pattern, cmd):
                return "dangerous", reason

    for cmd in (command, normalized):
        for pattern, reason in REQUIRE_APPROVAL_PATTERNS:
            if re.search(pattern, cmd):
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