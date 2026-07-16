"""后台进程注册表 — 管理通过 bash 工具启动的后台进程。

提供进程追踪、输出缓冲、状态查询和生命周期管理。
参考 Hermes Agent 的 tools/process_registry.py，简化适配 kocor 项目。
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from kocor.tools.toolsets.bash.constants import IS_WINDOWS, _find_bash

logger = logging.getLogger(__name__)

# 限制
MAX_OUTPUT_CHARS = 200_000  # 200KB 滚动输出缓冲
FINISHED_TTL_SECONDS = 1800  # 已结束进程保留 30 分钟
MAX_PROCESSES = 64  # 最大并发追踪进程数


@dataclass
class ProcessSession:
    """已追踪的后台进程，带输出缓冲。"""

    id: str  # 唯一会话 ID ("proc_xxxxxxxxxxxx")
    command: str  # 原始命令字符串
    pid: Optional[int] = None  # OS 进程 ID
    process: Optional[subprocess.Popen] = None  # Popen 句柄（本地模式）
    env_ref: Any = None  # 环境对象引用（sandbox 模式）
    cwd: Optional[str] = None  # 工作目录
    started_at: float = 0.0  # time.time() 启动时间
    exited: bool = False  # 是否已结束
    exit_code: Optional[int] = None  # 退出码
    completion_reason: str = "exited"  # exited/killed/failed_start
    output_buffer: str = ""  # 滚动输出缓冲
    max_output_chars: int = MAX_OUTPUT_CHARS
    pid_scope: str = "host"  # "host"（本地）/ "sandbox"（环境内）
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)


class ProcessRegistry:
    """后台进程注册表（模块级单例）。

    线程安全。提供进程的 spawn/poll/wait/kill/read_log/list 操作。
    """

    _SHELL_NOISE_SUBSTRINGS = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
    )

    def __init__(self):
        self._running: dict[str, ProcessSession] = {}
        self._finished: dict[str, ProcessSession] = {}
        self._lock = threading.Lock()
        self.completion_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def spawn(
        self,
        env: Any,
        command: str,
        *,
        cwd: str = "",
        timeout: int = 10,
        notify_on_complete: bool = False,
    ) -> ProcessSession:
        """通过环境接口启动后台进程（支持本地和 sandbox 后端）。"""
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            cwd=cwd,
            started_at=time.time(),
            env_ref=env,
            pid_scope="sandbox",
        )

        # 使用 nohup + & 在环境内后台执行
        temp_dir = "/tmp"
        log_path = f"{temp_dir}/kocor_bg_{session.id}.log"
        pid_path = f"{temp_dir}/kocor_bg_{session.id}.pid"
        exit_path = f"{temp_dir}/kocor_bg_{session.id}.exit"

        import shlex
        quoted_cmd = shlex.quote(command)
        bg_command = (
            f"mkdir -p {shlex.quote(temp_dir)} && "
            f"( nohup bash -lc {quoted_cmd} > {shlex.quote(log_path)} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)} ) & "
            f"echo $! > {shlex.quote(pid_path)} && cat {shlex.quote(pid_path)}"
        )

        try:
            result = env.execute(bg_command, timeout=timeout)
            output = result.get("stdout", "").strip()
            for line in output.splitlines():
                line = line.strip()
                if line.isdigit():
                    session.pid = int(line)
                    break
            if session.pid is None:
                session.exited = True
                session.exit_code = int(result.get("exit_code", -1))
                session.completion_reason = "failed_start"
                session.output_buffer = result.get("stdout", "")
        except Exception as e:
            session.exited = True
            session.exit_code = -1
            session.completion_reason = "failed_start"
            session.output_buffer = f"Failed to start: {e}"

        if not session.exited:
            reader = threading.Thread(
                target=self._env_poller_loop,
                args=(session, env, log_path, pid_path, exit_path),
                daemon=True,
                name=f"proc-poller-{session.id}",
            )
            session._reader_thread = reader
            reader.start()

        with self._lock:
            self._prune_if_needed()
            if not session.exited:
                self._running[session.id] = session

        return session

    def spawn_local(
        self,
        command: str,
        *,
        cwd: str = "",
        env_vars: Optional[dict] = None,
    ) -> ProcessSession:
        """通过本地 Popen 启动后台进程。

        仅用于本地执行模式。
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            cwd=cwd or os.getcwd(),
            started_at=time.time(),
        )

        # 使用用户 shell 启动后台进程
        user_shell = _find_bash()
        bg_env = os.environ.copy()
        if env_vars:
            bg_env.update(env_vars)
        bg_env["PYTHONUNBUFFERED"] = "1"

        _popen_kwargs = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if IS_WINDOWS else {}
        )

        proc = subprocess.Popen(
            [user_shell, "-lic", f"set +m; {command}"],
            text=True,
            cwd=session.cwd,
            env=bg_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            **_popen_kwargs,
        )

        session.process = proc
        session.pid = proc.pid

        reader = threading.Thread(
            target=self._reader_loop,
            args=(session,),
            daemon=True,
            name=f"proc-reader-{session.id}",
        )
        session._reader_thread = reader
        reader.start()

        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session

        return session

    # ------------------------------------------------------------------
    # Reader / Poller 线程
    # ------------------------------------------------------------------

    def _reader_loop(self, session: ProcessSession):
        """后台 reader 线程：增量读取本地 Popen 的 stdout。"""
        try:
            stdout = session.process.stdout
            if stdout is None:
                return
            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)
            while not session.exited:
                if raw_read is not None:
                    raw = raw_read(4096)
                    if not raw:
                        break
                    chunk = raw.decode("utf-8", errors="replace")
                else:
                    chunk = stdout.read(4096)
                    if not chunk:
                        break
                with session._lock:
                    session.output_buffer += chunk
                    if len(session.output_buffer) > session.max_output_chars:
                        session.output_buffer = session.output_buffer[-session.max_output_chars:]
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
            try:
                session.process.wait(timeout=5)
            except Exception:
                pass
            session.exited = True
            if session.completion_reason != "killed":
                session.exit_code = session.process.returncode
                session.completion_reason = "exited"
            self._move_to_finished(session)

    def _env_poller_loop(
        self,
        session: ProcessSession,
        env: Any,
        log_path: str,
        pid_path: str,
        exit_path: str,
    ):
        """后台 poller 线程：轮询环境内的日志文件（sandbox 后端）。"""
        import shlex
        quoted_log = shlex.quote(log_path)
        quoted_pid = shlex.quote(pid_path)
        quoted_exit = shlex.quote(exit_path)
        prev_output_len = 0

        while not session.exited:
            time.sleep(2)
            try:
                result = env.execute(f"cat {quoted_log} 2>/dev/null", timeout=10)
                new_output = result.get("stdout", "")
                if new_output:
                    _ = new_output[prev_output_len:]  # 计算增量，用于后续处理
                    prev_output_len = len(new_output)
                    with session._lock:
                        session.output_buffer = new_output
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]

                check = env.execute(
                    f"kill -0 \"$(cat {quoted_pid} 2>/dev/null)\" 2>/dev/null; echo $?",
                    timeout=5,
                )
                check_out = check.get("stdout", "").strip()
                if check_out and check_out.splitlines()[-1].strip() != "0":
                    exit_result = env.execute(f"cat {quoted_exit} 2>/dev/null", timeout=5)
                    exit_str = exit_result.get("stdout", "").strip()
                    try:
                        session.exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        session.exit_code = -1
                    session.exited = True
                    if session.completion_reason != "killed":
                        session.completion_reason = "exited"
                    self._move_to_finished(session)
                    return

            except Exception:
                session.exited = True
                session.exit_code = -1
                session.completion_reason = "lost"
                self._move_to_finished(session)
                return

    def _move_to_finished(self, session: ProcessSession):
        """将会话从 running 移到 finished。幂等操作。"""
        with self._lock:
            _was_running = self._running.pop(session.id, None) is not None
            self._finished[session.id] = session
        session._completion_event.set()

    # ------------------------------------------------------------------
    # 通知
    # ------------------------------------------------------------------

    def _notify_completion(
        self,
        session_id: str,
        command: str,
        exit_code: Optional[int],
        completion_reason: str = "exited",
        output: str = "",
    ) -> None:
        """向完成队列推送通知。"""
        self.completion_queue.put({
            "type": "completion",
            "session_id": session_id,
            "command": command,
            "exit_code": exit_code,
            "completion_reason": completion_reason,
            "output": output[-2000:] if output else "",
        })

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """按 ID 获取会话（running 或 finished）。"""
        with self._lock:
            return self._running.get(session_id) or self._finished.get(session_id)

    def poll(self, session_id: str) -> dict:
        """检查后台进程状态并获取新输出。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        # 本地模式：检查子进程是否已结束但 reader 仍阻塞
        if session.process is not None and not session.exited:
            rc = session.process.poll()
            if rc is not None:
                session.exited = True
                if session.completion_reason != "killed":
                    session.exit_code = rc
                self._move_to_finished(session)

        with session._lock:
            output_preview = session.output_buffer[-1000:] if session.output_buffer else ""

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at),
            "output_preview": output_preview,
        }
        if session.exited:
            result["exit_code"] = session.exit_code
            result["completion_reason"] = session.completion_reason
        return result

    def read_log(self, session_id: str, offset: int = 0, limit: int = 200) -> dict:
        """读取完整的输出日志，支持按行分页。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        with session._lock:
            full_output = session.output_buffer

        lines = full_output.splitlines()
        total_lines = len(lines)

        if offset == 0 and limit > 0:
            selected = lines[-limit:]
        else:
            selected = lines[offset:offset + limit]

        return {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": total_lines,
            "showing": f"{len(selected)} lines",
        }

    def wait(self, session_id: str, timeout: Optional[int] = None) -> dict:
        """阻塞等待进程退出。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        effective_timeout = timeout or 180
        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            session = self.get(session_id)
            if session is None:
                return {"status": "not_found"}
            if session.exited:
                self._notify_completion(
                    session.id, session.command, session.exit_code,
                    session.completion_reason, session.output_buffer,
                )
                return {
                    "status": "exited",
                    "command": session.command,
                    "exit_code": session.exit_code,
                    "completion_reason": session.completion_reason,
                    "output": session.output_buffer[-2000:],
                }
            remaining = deadline - time.monotonic()
            session._completion_event.wait(timeout=min(1.0, remaining))

        return {
            "status": "timeout",
            "command": session.command,
            "output": session.output_buffer[-1000:],
            "note": f"Waited {effective_timeout}s, process still running",
        }

    def kill(self, session_id: str) -> dict:
        """杀死后台进程。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "exit_code": session.exit_code}

        try:
            if session.process:
                if IS_WINDOWS:
                    session.process.kill()
                else:
                    try:
                        pgid = os.getpgid(session.process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        # 等待退出
                        deadline = time.monotonic() + 1.0
                        while time.monotonic() < deadline:
                            try:
                                os.killpg(pgid, 0)
                            except ProcessLookupError:
                                break
                            time.sleep(0.05)
                    except (ProcessLookupError, OSError):
                        session.process.kill()
            elif session.env_ref and session.pid:
                session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
            else:
                return {"status": "error", "error": "No process handle available"}

            session.exited = True
            session.exit_code = -15  # SIGTERM
            session.completion_reason = "killed"
            self._move_to_finished(session)
            return {"status": "killed", "session_id": session.id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_sessions(self) -> list[dict]:
        """列出所有后台进程（running + finished）。"""
        with self._lock:
            all_sessions = list(self._running.values()) + list(self._finished.values())

        result = []
        for s in all_sessions:
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            if s.exited:
                entry["exit_code"] = s.exit_code
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def _prune_if_needed(self):
        """移除超期的 finished 会话。必须在持有 _lock 时调用。"""
        now = time.time()
        expired = [
            sid for sid, s in self._finished.items()
            if (now - s.started_at) > FINISHED_TTL_SECONDS
        ]
        for sid in expired:
            del self._finished[sid]

        total = len(self._running) + len(self._finished)
        if total >= MAX_PROCESSES and self._finished:
            oldest_id = min(self._finished, key=lambda sid: self._finished[sid].started_at)
            del self._finished[oldest_id]


# 模块级单例
process_registry = ProcessRegistry()