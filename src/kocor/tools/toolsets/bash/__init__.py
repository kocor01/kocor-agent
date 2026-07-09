"""bash 工具集子包。

包含：
- constants: Shell 查找、路径常量、环境构建
- environment: BaseEnvironment 抽象基类 + LocalEnvironment 实现
- command_safety: 危险命令检测、workdir 验证
- output: 输出后处理（ANSI 剥离、脱敏、截断）
- process_registry: 后台进程注册表
"""

from kocor.tools.toolsets.bash.constants import (
    IS_WINDOWS,
    _find_bash,
    _make_run_env,
    _resolve_safe_cwd,
    _quote_cwd_for_cd,
)
from kocor.tools.toolsets.bash.environment import (
    BaseEnvironment,
    LocalEnvironment,
)
from kocor.tools.toolsets.bash.command_safety import (
    detect_dangerous_command,
    validate_workdir,
)
from kocor.tools.toolsets.bash.output import (
    OutputProcessor,
    strip_ansi,
    truncate_output,
    redact_sensitive,
)
from kocor.tools.toolsets.bash.process_registry import (
    ProcessRegistry,
    ProcessSession,
    process_registry,
)

__all__ = [
    "IS_WINDOWS",
    "_find_bash",
    "_make_run_env",
    "_resolve_safe_cwd",
    "_quote_cwd_for_cd",
    "BaseEnvironment",
    "LocalEnvironment",
    "detect_dangerous_command",
    "validate_workdir",
    "OutputProcessor",
    "strip_ansi",
    "truncate_output",
    "redact_sensitive",
    "ProcessRegistry",
    "ProcessSession",
    "process_registry",
]