"""命令安全审查模块：危险命令检测、workdir 验证。

提供三层防护：
1. shlex 语义解析层 — 准确识别命令名和参数，避免正则误匹配和安全绕过
2. 正则匹配层 — 兜底捕获 shlex 无法解析的混淆命令
3. 命令规范化 — 在检测前处理常见的 shell 混淆绕过（引号分段等）

规范化只用于模式匹配检测，实际执行的命令不变。
"""

import re
import shlex
from typing import Optional

# ---------------------------------------------------------------------------
# shlex 语义解析：准确识别命令名和参数
# ---------------------------------------------------------------------------

# 已知危险命令：命令名一经出现即视为 dangerous
_DANGEROUS_COMMANDS: frozenset[str] = frozenset({
    "mkfs", "mkfs.ext2", "mkfs.ext3", "mkfs.ext4", "mkfs.btrfs",
    "mkfs.xfs", "mkfs.fat", "mkfs.ntfs",
    "dd", "usermod", "passwd", "xmrig", "minerd", "cryptominer",
})

# 已知危险命令及触发危险级别的参数组合
# 注意：rm 的危险级别（rm -rf /）由 regex 层处理（需检查目标路径是否为 /），
# shlex 层不包含 rm 的 dangerous 规则，避免误判 rm -rf ./node_modules。
_DANGEROUS_COMMAND_RULES: dict[str, list[tuple[list[str], str]]] = {
}

# 已知需审批命令及触发 caution 的参数组合
# 空 trigger_args（如 "rm": [([], "…")]）表示任何该命令调用都触发 caution。
# 若需对特定参数跳过检查（如 sudo -S），在 _SAFE_SKIP_FLAGS 中定义。
_REQUIRE_APPROVAL_COMMANDS: dict[str, list[tuple[list[str], str]]] = {
    "rm": [
        ([], "Approval needed: rm command"),
    ],
    "chmod": [
        (["-R", "--recursive"], "Approval needed: recursive chmod"),
        ([], "Approval needed: file permission change"),
    ],
    "chown": [
        (["-R", "--recursive"], "Approval needed: recursive chown"),
    ],
    "kill": [
        ([], "Approval needed: kill process"),
    ],
    "killall": [
        ([], "Approval needed: killall process"),
    ],
    "pkill": [
        ([], "Approval needed: pkill process"),
    ],
    "nohup": [
        ([], "Approval needed: nohup background"),
    ],
    "wget": [
        ([], "Approval needed: wget download"),
    ],
    "sudo": [
        ([], "Approval needed: sudo (use -S flag if needed)"),
    ],
    "eval": [
        ([], "Approval needed: eval (possible safety bypass)"),
    ],
    "find": [
        (["-exec"], "Approval needed: find with exec"),
        (["-delete"], "Approval needed: find with delete"),
    ],
}

# 安全跳过标记：某些命令带特定参数时无需 caution
# 如 sudo -S（从 stdin 读密码）是安全用法
_SAFE_SKIP_FLAGS: dict[str, frozenset[str]] = {
    "sudo": frozenset({"-S"}),
}


def _parse_command_tokens(command: str) -> list[str]:
    """将命令解析为 token 列表，失败时返回空列表。"""
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _check_tokens(tokens: list[str]) -> tuple[str, str]:
    """基于 token 语义检查命令安全性。

    Returns:
        ("safe"/"caution"/"dangerous", 原因描述或 "")
    """
    if not tokens:
        return "safe", ""

    cmd_name = tokens[0]
    args = tokens[1:]

    # 危险命令（命令名匹配即危险）
    if cmd_name in _DANGEROUS_COMMANDS:
        return "dangerous", f"Dangerous: {cmd_name} command"

    # 危险命令规则（参数匹配）
    if cmd_name in _DANGEROUS_COMMAND_RULES:
        rules = _DANGEROUS_COMMAND_RULES[cmd_name]
        for trigger_args, reason in rules:
            if not trigger_args:
                return "dangerous", reason
            if any(ta in args for ta in trigger_args):
                return "dangerous", reason
        # 命令名匹配但无危险参数 → 降级 caution
        return "caution", f"Approval needed: {cmd_name} command"

    # 安全跳过检查：某些命令带特定参数时无需 caution
    # 如 sudo -S（从 stdin 读密码）是安全用法
    if cmd_name in _SAFE_SKIP_FLAGS:
        if any(flag in args for flag in _SAFE_SKIP_FLAGS[cmd_name]):
            return "safe", ""

    # 需审批命令规则
    if cmd_name in _REQUIRE_APPROVAL_COMMANDS:
        rules = _REQUIRE_APPROVAL_COMMANDS[cmd_name]
        for trigger_args, reason in rules:
            if not trigger_args or any(ta in args for ta in trigger_args):
                return "caution", reason

    # 间接 shell 调用检测
    if cmd_name in ("sh", "bash", "zsh", "dash", "ksh"):
        return "caution", f"Approval needed: {cmd_name} shell invocation"

    return "safe", ""

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
    # 危险工具通过 xargs 间接执行
    (r"\bxargs\s+(?:rm|dd|mkfs|chmod|chown|mv)\b", "Dangerous: xargs with destructive command"),
    # 通过变量赋值隐藏危险工具（如 x=rm;$x -rf /）
    (r"\b\w+=(?:rm|dd|mkfs|chmod|chown|sudo)\b.*\$\{?\w+\}?", "Dangerous: variable masking dangerous tool"),
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
    # eval 可用于绕过所有安全检查
    (r"\beval\s", "Approval needed: eval (possible safety bypass)"),
    # find 递归执行/删除
    (r"\bfind\s+.*\s+-exec\s", "Approval needed: find with exec"),
    (r"\bfind\s+.*\s+-delete\b", "Approval needed: find with delete"),
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

    三层防护：
    1. 正则 dangerous 层 — 最高优先级，直接命中危险模式（如 rm -rf /）
    2. shlex 语义解析层 — 准确识别命令名和参数
    3. 正则 caution 层 — 兜底捕获需审批的命令

    Returns:
        ("safe"/"caution"/"dangerous", 原因描述或 "")
    """
    if not command:
        return "safe", ""

    normalized = _normalize_command(command)

    # 1. 正则 dangerous 层（最高优先级）：直接命中危险模式
    for cmd in (command, normalized):
        for pattern, reason in DANGEROUS_PATTERNS:
            if re.search(pattern, cmd):
                return "dangerous", reason

    # 2. shlex 语义解析层：准确识别命令名和参数，防遗漏
    tokens = _parse_command_tokens(command)
    if tokens:
        level, reason = _check_tokens(tokens)
        if level != "safe":
            return level, reason

    # 3. 正则 caution 层：兜底捕获需审批的命令
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