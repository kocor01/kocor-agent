"""Bash 工具和后台进程管理工具。

提供两个 LLM 可调用的工具：
- bash: 在本地 shell 中执行命令（前台/后台）
- process: 管理后台进程（poll/wait/kill/log/list）

核心代码在 bash/ 子包中。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from kocor.tools.permission import PermissionManager
from kocor.tools.toolset.bash.command_safety import (
    detect_dangerous_command,
    validate_workdir,
)
from kocor.tools.toolset.bash.environment import LocalEnvironment
from kocor.tools.toolset.bash.output import OutputProcessor
from kocor.tools.toolset.bash.process_registry import process_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局单例环境（跨调用持久化 CWD 和 shell 状态）
# ---------------------------------------------------------------------------

_env: Optional[LocalEnvironment] = None


def _get_env() -> LocalEnvironment:
    """获取或创建 LocalEnvironment 单例。"""
    global _env
    if _env is None:
        _env = LocalEnvironment(cwd=os.getcwd(), timeout=180)
    return _env


def _reset_env() -> None:
    """重置环境（清理快照文件并重建）。"""
    global _env
    if _env is not None:
        _env.cleanup()
    _env = LocalEnvironment(cwd=os.getcwd(), timeout=180)


# ---------------------------------------------------------------------------
# BashTool
# ---------------------------------------------------------------------------


class BashTool:
    """在本地 shell 中执行命令。

    CWD 和导出的环境变量在调用之间持久化。
    支持前台（默认）和后台两种执行模式。
    """

    NAME = "bash"
    DESCRIPTION = (
        "Execute shell commands on a Linux/Unix environment. "
        "The current working directory (CWD) and exported environment variables "
        "persist between calls.\n\n"
        "Foreground (default): Commands return when done. "
        "Use timeout for long-running commands.\n"
        "Background: Set background=true to get a session_id for later queries. "
        "Use process(action='poll') to check status, "
        "process(action='wait') to block until done.\n\n"
        "Do NOT use cat/head/tail to read files -- use the read_file tool instead.\n"
        "Do NOT use sed/awk to edit files -- use the write_file tool instead.\n"
        "Reserve terminal for: builds, installs, git, processes, scripts, network."
    )
    SAFETY_LEVEL = PermissionManager.SAFETY_DANGEROUS
    PARAMETERS = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "background": {
                "type": "boolean",
                "description": "Run in background (returns session_id for later query)",
                "default": False,
            },
            "workdir": {
                "type": "string",
                "description": "Working directory (optional, uses tracked CWD by default)",
            },
            "timeout": {
                "type": "integer",
                "description": "Foreground timeout in seconds (default 180)",
                "default": 180,
            },
        },
        "required": ["command"],
    }

    @staticmethod
    def handler(
        command: str,
        background: bool = False,
        workdir: str = "",
        timeout: int = 180,
    ) -> str:
        # 安全检查：空命令
        if not command:
            return json.dumps({"error": "Empty command"}, ensure_ascii=False)

        # 安全检查：危险命令检测
        level, reason = detect_dangerous_command(command)
        if level == "dangerous":
            return json.dumps({
                "error": "Command blocked",
                "reason": reason,
                "command": command,
            }, ensure_ascii=False)

        # 安全检查：workdir 字符白名单
        workdir_err = validate_workdir(workdir)
        if workdir_err:
            return json.dumps({
                "error": workdir_err,
                "workdir": workdir,
            }, ensure_ascii=False)

        if background:
            return BashTool._handle_background(command, workdir)
        else:
            return BashTool._handle_foreground(command, workdir, timeout)

    @staticmethod
    def _handle_foreground(command: str, workdir: str, timeout: int) -> str:
        """处理前台命令执行。"""
        env = _get_env()
        try:
            result = env.execute(command, cwd=workdir, timeout=timeout)
            stdout = result.get("stdout", "")
            exit_code = result.get("exit_code", 0)

            # 输出后处理
            processed = OutputProcessor.process(stdout)

            # 构建返回结果
            response = {
                "output": processed,
                "exit_code": exit_code,
            }
            if result.get("exit_code_note"):
                response["exit_code_note"] = result["exit_code_note"]
            if result.get("timed_out"):
                response["timed_out"] = True

            return json.dumps(response, ensure_ascii=False)

        except Exception as e:
            logger.exception("Bash foreground execution failed")
            return json.dumps({
                "error": f"Execution failed: {type(e).__name__}: {e}",
            }, ensure_ascii=False)

    @staticmethod
    def _handle_background(command: str, workdir: str) -> str:
        """处理后台上进程启动。"""
        env = _get_env()
        try:
            session = process_registry.spawn(
                env,
                command,
                cwd=workdir or env.cwd,
            )
            return json.dumps({
                "session_id": session.id,
                "command": session.command,
                "status": "running",
                "pid": session.pid,
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception("Bash background spawn failed")
            return json.dumps({
                "error": f"Background spawn failed: {type(e).__name__}: {e}",
            }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ProcessTool
# ---------------------------------------------------------------------------


class ProcessTool:
    """管理 bash 后台进程。

    提供以下操作：
    - list: 列出所有进程
    - poll: 检查状态并获取新输出
    - log: 读取完整输出日志（支持分页）
    - wait: 阻塞等待进程完成
    - kill: 终止进程
    """

    NAME = "process"
    DESCRIPTION = (
        "Manage background processes started with bash(background=true). "
        "Actions: 'list' (show all), 'poll' (check status + new output), "
        "'log' (full output with pagination), 'wait' (block until done), "
        "'kill' (terminate)."
    )
    SAFETY_LEVEL = PermissionManager.SAFETY_CAUTION
    PARAMETERS = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill"],
                "description": "Action to perform on background processes",
            },
            "session_id": {
                "type": "string",
                "description": "Process session ID. Required for all actions except 'list'.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to block for 'wait' action",
                "minimum": 1,
            },
            "offset": {
                "type": "integer",
                "description": "Line offset for 'log' action (default: last 200 lines)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return for 'log' action",
                "minimum": 1,
            },
        },
        "required": ["action"],
    }

    @staticmethod
    def handler(
        action: str,
        session_id: str = "",
        timeout: Optional[int] = None,
        offset: int = 0,
        limit: int = 200,
    ) -> str:
        if action == "list":
            sessions = process_registry.list_sessions()
            if not sessions:
                return json.dumps({"processes": [], "message": "No background processes"}, ensure_ascii=False)
            return json.dumps({"processes": sessions}, ensure_ascii=False)

        if not session_id:
            return json.dumps({"error": "session_id is required"}, ensure_ascii=False)

        if action == "poll":
            return json.dumps(process_registry.poll(session_id), ensure_ascii=False)
        elif action == "log":
            return json.dumps(
                process_registry.read_log(session_id, offset=offset, limit=limit),
                ensure_ascii=False,
            )
        elif action == "wait":
            return json.dumps(
                process_registry.wait(session_id, timeout=timeout),
                ensure_ascii=False,
            )
        elif action == "kill":
            return json.dumps(process_registry.kill(session_id), ensure_ascii=False)
        else:
            return json.dumps(
                {"error": f"Unknown action: {action}. Use: list, poll, log, wait, kill"},
                ensure_ascii=False,
            )