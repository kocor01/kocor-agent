"""执行环境模块：BaseEnvironment 抽象基类 + LocalEnvironment 本地实现。

架构参考 Hermes Agent 的 tools/environments/base.py 和 tools/environments/local.py，
但简化适配 kocor 项目"小而美"的定位，仅支持本地 subprocess 执行。
"""

from __future__ import annotations

import codecs
import logging
import os
import re
import select
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import IO

from kocor.tools.toolsets.bash.constants import (
    IS_WINDOWS,
    _find_bash,
    _make_run_env,
    _quote_cwd_for_cd,
    _resolve_safe_cwd,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSYS / Windows 路径转换
# ---------------------------------------------------------------------------


def _msys_to_windows_path(cwd: str) -> str:
    """将 MSYS 风格 POSIX 路径（/c/Users/x）转换为 Windows 原生格式（C:\\Users\\x）。

    非 Windows 平台或非 MSYS 格式路径原样返回。
    """
    if not IS_WINDOWS or not cwd:
        return cwd
    m = re.match(r'^/([a-zA-Z])(/.*)?$', cwd)
    if not m:
        return cwd
    drive = m.group(1).upper()
    tail = (m.group(2) or "").replace('/', '\\')
    return f"{drive}:{tail}"


def _windows_to_msys_path(cwd: str) -> str:
    """将 Windows 原生路径（C:\\Users\\x）转换为 MSYS 风格（/c/Users/x）。

    非 Windows 平台或非 Windows 路径原样返回。
    """
    if not IS_WINDOWS or not cwd:
        return cwd
    m = re.match(r'^([a-zA-Z]):[\\/]*(.*)$', cwd)
    if not m:
        return cwd
    drive = m.group(1).lower()
    tail = m.group(2).replace('\\', '/').lstrip('/')
    return f"/{drive}/{tail}" if tail else f"/{drive}/"


# ---------------------------------------------------------------------------
# ProcessHandle protocol
# ---------------------------------------------------------------------------


class ProcessHandle:
    """进程句柄协议，适配 subprocess.Popen 和测试 mock。

    提供统一的 poll() / kill() / wait() / stdout 接口。
    """

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    @property
    def stdout(self) -> IO[str] | None: ...
    @property
    def returncode(self) -> int | None: ...


# ---------------------------------------------------------------------------
# 管道写入辅助
# ---------------------------------------------------------------------------


def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """在守护线程中向 proc.stdin 写入数据，避免管道缓冲区死锁。"""

    def _write():
        try:
            raw = data.encode("utf-8") if isinstance(data, str) else data
            target = getattr(proc.stdin, "buffer", proc.stdin)
            target.write(raw)
            target.close()
        except (BrokenPipeError, OSError):
            pass

    threading.Thread(target=_write, daemon=True).start()


# ---------------------------------------------------------------------------
# 退出码解释
# ---------------------------------------------------------------------------

_EXIT_CODE_SEMANTICS: dict[str, dict[int, str]] = {
    "grep": {1: "No matches found (not an error)"},
    "egrep": {1: "No matches found (not an error)"},
    "fgrep": {1: "No matches found (not an error)"},
    "rg": {1: "No matches found (not an error)"},
    "diff": {1: "Files differ (expected, not an error)"},
    "find": {1: "Some directories were inaccessible (partial results may still be valid)"},
    "test": {1: "Condition evaluated to false (expected, not an error)"},
    "[": {1: "Condition evaluated to false (expected, not an error)"},
    "curl": {6: "Could not resolve host", 7: "Failed to connect to host"},
}


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """返回非零退出码的解释（如 grep 返回 1 表示"未找到"）。"""
    if exit_code == 0:
        return None
    # 取最后一个命令段
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue
        base_cmd = w.split("/")[-1]
        break
    if not base_cmd:
        return None
    cmd_semantics = _EXIT_CODE_SEMANTICS.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]
    return None


# ---------------------------------------------------------------------------
# BaseEnvironment
# ---------------------------------------------------------------------------


class BaseEnvironment(ABC):
    """执行环境基类。

    职责：
    - 统一的 execute() 接口
    - 会话快照（环境变量持久化）
    - CWD 追踪
    - 命令包装（source snapshot + cd + eval + CWD marker）
    """

    def __init__(self, cwd: str, timeout: int):
        self.cwd = cwd
        self.timeout = timeout
        self._session_id = uuid.uuid4().hex[:12]
        temp_dir = self.get_temp_dir().rstrip("/") or "/"
        self._snapshot_path = f"{temp_dir}/kocor-snap-{self._session_id}.sh"
        self._cwd_file = f"{temp_dir}/kocor-cwd-{self._session_id}.txt"
        self._cwd_marker = f"__KOCOR_CWD_{self._session_id}__"
        self._snapshot_ready = False

    def get_temp_dir(self) -> str:
        """返回临时目录路径。"""
        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"
        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"
        candidate = tempfile.gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"
        return "/tmp"

    # ------------------------------------------------------------------
    # 抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        """启动 bash 子进程执行命令。返回 ProcessHandle 兼容对象。"""

    @abstractmethod
    def cleanup(self):
        """释放后端资源。"""

    # ------------------------------------------------------------------
    # 会话快照
    # ------------------------------------------------------------------

    def init_session(self):
        """捕获登录 shell 环境到快照文件。

        快照在首次构造时创建一次，后续命令 source 快照而非每次重新加载 profile。
        """
        _quoted_cwd = _quote_cwd_for_cd(self.cwd)
        _quoted_snap = shlex.quote(self._snapshot_path)
        _quoted_cwd_file = shlex.quote(self._cwd_file)
        _snap_tmp = shlex.quote(self._snapshot_path + ".tmp.") + "$BASHPID"

        bootstrap = (
            f"export -p > {_snap_tmp}\n"
            f"__kocor_fns=$(declare -F | awk '{{print $3}}' | grep -vE '^_[^_]') || true\n"
            f"[ -n \"$__kocor_fns\" ] && declare -f $__kocor_fns >> {_snap_tmp} 2>/dev/null || true\n"
            f"alias -p >> {_snap_tmp}\n"
            f"echo 'shopt -s expand_aliases' >> {_snap_tmp}\n"
            f"echo 'set +e' >> {_snap_tmp}\n"
            f"echo 'set +u' >> {_snap_tmp}\n"
            f"mv -f {_snap_tmp} {_quoted_snap} || rm -f {_snap_tmp}\n"
            f"builtin cd -- {_quoted_cwd} 2>/dev/null || true\n"
            f"pwd -P > {_quoted_cwd_file} 2>/dev/null || true\n"
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\"\n"
        )
        try:
            proc = self._run_bash(bootstrap, login=True, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            if int(result.get("exit_code") or 0) != 0:
                raise RuntimeError(
                    f"snapshot bootstrap failed with exit code {result.get('exit_code')}"
                )
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info(
                "Session snapshot created (session=%s, cwd=%s)",
                self._session_id, self.cwd,
            )
        except Exception as exc:
            logger.warning(
                "init_session failed (session=%s): %s -- falling back to bash -l per command",
                self._session_id, exc,
            )
            self._snapshot_ready = False

    _snapshot_timeout: int = 30

    # ------------------------------------------------------------------
    # 命令包装
    # ------------------------------------------------------------------

    def _wrap_command(self, command: str, cwd: str) -> str:
        """构建完整的 bash 脚本：source 快照 → cd → eval → 重新导出 → CWD 标记。

        输出示例：
            source /tmp/kocor-snap-xxx.sh >/dev/null 2>&1 || true
            builtin cd -- /home/user/project || exit 126
            eval 'python3 -m http.server 8080'
            __kocor_ec=$?
            { export -p > /tmp/kocor-snap-xxx.sh.tmp.$BASHPID && \\
              mv -f /tmp/kocor-snap-xxx.sh.tmp.$BASHPID /tmp/kocor-snap-xxx.sh; } \\
              2>/dev/null || rm -f /tmp/kocor-snap-xxx.sh.tmp.$BASHPID 2>/dev/null || true
            pwd -P > /tmp/kocor-cwd-xxx.txt 2>/dev/null || true
            printf '\\n__KOCOR_CWD_xxx__%s__KOCOR_CWD_xxx__\\n' "$(pwd -P)"
            exit $__kocor_ec
        """
        escaped = command.replace("'", "'\\''")
        _quoted_snap = shlex.quote(self._snapshot_path)
        _quoted_cwd_file = shlex.quote(self._cwd_file)
        _snap_tmp = shlex.quote(self._snapshot_path + ".tmp.") + "$BASHPID"

        parts = []
        if self._snapshot_ready:
            parts.append(f"source {_quoted_snap} >/dev/null 2>&1 || true")

        quoted_cwd = _quote_cwd_for_cd(cwd)
        parts.append(f"builtin cd -- {quoted_cwd} || exit 126")
        parts.append(f"eval '{escaped}'")
        parts.append("__kocor_ec=$?")

        if self._snapshot_ready:
            parts.append(
                f"{{ export -p > {_snap_tmp} && mv -f {_snap_tmp} {_quoted_snap}; }} "
                f"2>/dev/null || rm -f {_snap_tmp} 2>/dev/null || true"
            )

        parts.append(f"pwd -P > {_quoted_cwd_file} 2>/dev/null || true")
        parts.append(
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\""
        )
        parts.append("exit $__kocor_ec")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 进程生命周期
    # ------------------------------------------------------------------

    def _wait_for_process(self, proc: subprocess.Popen, timeout: int = 120) -> dict:
        """基于轮询的进程等待，支持中断检查和 stdout 读取。

        Returns:
            {"stdout": str, "exit_code": int}
        """
        output_chunks: list[str] = []
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _drain():
            stream = proc.stdout
            if stream is None:
                return
            fileno = getattr(stream, "fileno", None)
            try:
                fd = fileno() if callable(fileno) else None
            except Exception:
                fd = None
            if not isinstance(fd, int) or fd < 0:
                # 非真实 fd 的 fallback 路径
                try:
                    for piece in stream:
                        if piece is None:
                            continue
                        if isinstance(piece, bytes):
                            output_chunks.append(decoder.decode(piece))
                        else:
                            output_chunks.append(str(piece))
                except Exception:
                    pass
                finally:
                    try:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            output_chunks.append(tail)
                    except Exception:
                        pass
                return
            # 真实 fd 路径：select() 非阻塞读取
            if os.name == "nt":
                try:
                    while True:
                        chunk = os.read(fd, 4096)
                        if not chunk:
                            break
                        output_chunks.append(decoder.decode(chunk))
                except (ValueError, OSError):
                    pass
                finally:
                    try:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            output_chunks.append(tail)
                    except Exception:
                        pass
                return
            idle_after_exit = 0
            try:
                while True:
                    try:
                        ready, _, _ = select.select([fd], [], [], 0.1)
                    except (ValueError, OSError):
                        break
                    if ready:
                        try:
                            chunk = os.read(fd, 4096)
                        except (ValueError, OSError):
                            break
                        if not chunk:
                            break
                        output_chunks.append(decoder.decode(chunk))
                        idle_after_exit = 0
                    elif proc.poll() is not None:
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
            finally:
                try:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        output_chunks.append(tail)
                except Exception:
                    pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        deadline = time.monotonic() + timeout

        try:
            _poll_sleep = 0.005
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    self._kill_process(proc)
                    drain_thread.join(timeout=2)
                    partial = "".join(output_chunks)
                    timeout_msg = f"\n[Command timed out after {timeout}s]"
                    return {
                        "stdout": partial + timeout_msg if partial else timeout_msg.lstrip(),
                        "exit_code": 124,
                        "timed_out": True,
                    }
                time.sleep(_poll_sleep)
                if _poll_sleep < 0.2:
                    _poll_sleep = min(_poll_sleep * 1.5, 0.2)
        except (KeyboardInterrupt, SystemExit):
            self._kill_process(proc)
            drain_thread.join(timeout=2)
            return {
                "stdout": "".join(output_chunks) + "\n[Command interrupted]",
                "exit_code": 130,
                "interrupted": True,
            }

        drain_thread.join(timeout=2)
        try:
            proc.stdout.close()
        except Exception:
            pass

        exit_code = proc.returncode if proc.returncode is not None else 0
        result = {
            "stdout": "".join(output_chunks),
            "exit_code": exit_code,
        }
        return result

    def _kill_process(self, proc: subprocess.Popen):
        """终止进程。子类可覆盖以支持进程组杀死。"""
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # ------------------------------------------------------------------
    # CWD 提取
    # ------------------------------------------------------------------

    def _update_cwd(self, result: dict):
        """从命令输出中提取 CWD。"""
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """从 stdout 中解析 __KOCOR_CWD_{session}__ 标记。"""
        output = result.get("stdout", "")
        marker = self._cwd_marker
        last = output.rfind(marker)
        if last == -1:
            return
        search_start = max(0, last - 4096)
        first = output.rfind(marker, search_start, last)
        if first == -1 or first == last:
            return
        cwd_path = output[first + len(marker):last].strip()
        if cwd_path:
            self.cwd = cwd_path
        # 从输出中剥离标记行
        line_start = output.rfind("\n", 0, first)
        if line_start == -1:
            line_start = first
        line_end = output.find("\n", last + len(marker))
        line_end = line_end + 1 if line_end != -1 else len(output)
        result["stdout"] = output[:line_start] + output[line_end:]

    # ------------------------------------------------------------------
    # 统一 execute()
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        """执行命令，返回 {"stdout": str, "exit_code": int}。

        包含 execute 级别的防御纵深安全检查（不替代上层 PermissionManager）。
        危险命令（如 rm -rf /）直接抛出 PermissionError。
        """
        if not command:
            return {"stdout": "Error: empty command", "exit_code": 1}

        # 防御纵深：execute 层安全检查（不替代 PermissionManager）
        from kocor.tools.toolsets.bash.command_safety import detect_dangerous_command
        level, reason = detect_dangerous_command(command)
        if level == "dangerous":
            raise PermissionError(f"Blocked by execute-level safety check: {reason}")

        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd
        # 在 wrap 之前确保 CWD 存在（否则 cd 会失败）
        effective_cwd = _resolve_safe_cwd(effective_cwd)

        wrapped = self._wrap_command(command, effective_cwd)
        login = not self._snapshot_ready

        proc = self._run_bash(
            wrapped, login=login, timeout=effective_timeout, stdin_data=stdin_data,
        )
        result = self._wait_for_process(proc, timeout=effective_timeout)
        if result["exit_code"] is not None and result["exit_code"] != 0:
            note = _interpret_exit_code(command, result["exit_code"])
            if note:
                result["exit_code_note"] = note
        self._update_cwd(result)
        return result

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def stop(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# =============================================================================
# LocalEnvironment
# =============================================================================


class LocalEnvironment(BaseEnvironment):
    """本地 subprocess.Popen 执行环境。

    每个 execute() 调用启动一个全新的 bash -c 进程。
    会话快照在构造时创建，环境变量跨命令持久化。
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict | None = None):
        if cwd:
            cwd = os.path.expanduser(cwd)
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout)
        self._env = env or {}
        self.init_session()

    def get_temp_dir(self) -> str:
        """返回 shell-safe 的临时目录。"""
        if IS_WINDOWS:
            cache_dir = os.path.join(tempfile.gettempdir(), "kocor_terminal")
            os.makedirs(cache_dir, exist_ok=True)
            return cache_dir.replace("\\", "/")
        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"
        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"
        candidate = tempfile.gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"
        return "/tmp"

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        bash = _find_bash()
        args = [bash, "-l", "-c", cmd_string] if login else [bash, "-c", cmd_string]
        run_env = _make_run_env()
        safe_cwd = _resolve_safe_cwd(self.cwd)

        if safe_cwd != self.cwd:
            logger.warning(
                "LocalEnvironment cwd %r is missing on disk; falling back to %r",
                self.cwd, safe_cwd,
            )
            self.cwd = safe_cwd

        _popen_kwargs = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if IS_WINDOWS
            else {}
        )

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True,
            cwd=self.cwd,
            **_popen_kwargs,
        )

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def _kill_process(self, proc: subprocess.Popen):
        """杀死整个进程组（所有子进程）。"""
        try:
            if IS_WINDOWS:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                # 等待进程组退出
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        proc.poll()
                    except Exception:
                        pass
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        return
                    time.sleep(0.05)
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                try:
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def _update_cwd(self, result: dict):
        """本地环境：从文件读取 CWD，兼容 Windows MSYS 路径。"""
        try:
            with open(self._cwd_file, encoding="utf-8") as f:
                cwd_path = f.read().strip()
            if IS_WINDOWS:
                cwd_path = _msys_to_windows_path(cwd_path)
            if cwd_path and os.path.isdir(cwd_path):
                self.cwd = cwd_path
        except (OSError, FileNotFoundError):
            pass
        # 仍然从输出中剥离标记
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """从 stdout 提取 CWD，Windows 上转换 MSYS 路径为原生格式。"""
        prev_cwd = self.cwd
        super()._extract_cwd_from_output(result)
        if self.cwd != prev_cwd and IS_WINDOWS:
            normalized = _msys_to_windows_path(self.cwd)
            if normalized and os.path.isdir(normalized):
                self.cwd = normalized
            else:
                self.cwd = prev_cwd

    def cleanup(self):
        """清理临时文件。"""
        for f in (self._snapshot_path, self._cwd_file):
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass
        # 清理孤立的原子写入临时文件
        try:
            import glob
            for tmp in glob.glob(f"{self._snapshot_path}.tmp.*"):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass