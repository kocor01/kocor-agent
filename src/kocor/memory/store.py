"""双文件平铺长期记忆存储。

Hermes 风格的 MEMORY.md + USER.md 双文件存储：
- § 分隔的多行条目
- 字符上限约束
- 冻结快照（会话内不变，保前缀缓存命中）
- 子串匹配的 replace/remove
- 批量原子操作（按最终预算校验，全有或全无）
- 原子写入（temp file + os.replace）
- 跨平台文件锁（fcntl/msvcrt）
- strict scope 威胁模式扫描
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kocor.memory.threat_patterns import scan_strict
from kocor.memory.types import MemorySnapshot, MemoryTarget

# 条目分隔符
ENTRY_SEPARATOR = "\n§\n"

# 文件名
MEMORY_FILENAME = "MEMORY.md"
USER_FILENAME = "USER.md"


@dataclass
class MemoryOp:
    """单个记忆操作。"""

    action: Literal["add", "replace", "remove"]
    target: MemoryTarget
    content: str = ""
    old_substring: str = ""


@dataclass
class MemoryOpResult:
    """操作结果。"""

    success: bool
    target: MemoryTarget | None = None
    error: str = ""
    current_entries: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class MemoryStore:
    """双文件平铺记忆存储。"""

    def __init__(
        self,
        memory_dir: str,
        memory_limit: int,
        user_limit: int,
        user_enabled: bool,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_limit = memory_limit
        self.user_limit = user_limit
        self.user_enabled = user_enabled

        self._memory_path = self.memory_dir / MEMORY_FILENAME
        self._user_path = self.memory_dir / USER_FILENAME

        self._snapshot: MemorySnapshot = MemorySnapshot(
            memory_usage=(0, memory_limit),
            user_usage=(0, user_limit),
        )
        self._snapshot_version: int = 0
        self._last_snapshot_at: float = 0.0

    # ── 生命周期 ────────────────────────────────────────

    def load_from_disk(self) -> MemorySnapshot:
        """加载文件、去重、威胁扫描、生成冻结快照。"""
        for path in (self._memory_path, self._user_path):
            if not path.exists():
                path.write_text("", encoding="utf-8")

        memory_entries = self._read_and_dedup(self._memory_path)
        user_entries = self._read_and_dedup(self._user_path)

        memory_for_snapshot = [self._maybe_block(e) for e in memory_entries]
        user_for_snapshot = [self._maybe_block(e) for e in user_entries]

        memory_used = self._calc_usage(memory_for_snapshot)
        user_used = self._calc_usage(user_for_snapshot)

        formatted = self._format_snapshot_text(
            memory_for_snapshot, user_for_snapshot,
            (memory_used, self.memory_limit), (user_used, self.user_limit),
        )

        self._snapshot = MemorySnapshot(
            memory_entries=memory_for_snapshot,
            user_entries=user_for_snapshot,
            memory_usage=(memory_used, self.memory_limit),
            user_usage=(user_used, self.user_limit),
            formatted_text=formatted,
        )
        self._snapshot_version += 1
        self._last_snapshot_at = time.time()
        return self._snapshot

    def refresh_snapshot(self) -> MemorySnapshot:
        """从磁盘重新加载，生成新快照。

        在 add/replace/remove 操作后调用，使新记忆对下一轮 LLM 可见。
        不改变快照结构（标题/格式），仅更新条目文本，
        因此 system prompt 的前缀缓存仍有效。
        """
        return self.load_from_disk()

    @property
    def snapshot_version(self) -> int:
        """快照版本号，每次 refresh 递增。"""
        return self._snapshot_version

    def format_for_system_prompt(self) -> str:
        """返回冻结快照的 formatted_text（会话内不变）。"""
        return self._snapshot.formatted_text

    def list_entries(self, target: MemoryTarget) -> list[str]:
        """从磁盘读取当前条目列表（实时状态，非快照）。"""
        path = self._path_of(target)
        if not path.exists():
            return []
        return self._read_and_dedup(path)

    # ── 单操作 ────────────────────────────────────────

    def add(self, target: MemoryTarget, content: str) -> MemoryOpResult:
        return self._apply_single(MemoryOp(action="add", target=target, content=content))

    def replace(self, target: MemoryTarget, old_substring: str, new_content: str) -> MemoryOpResult:
        return self._apply_single(
            MemoryOp(action="replace", target=target, content=new_content, old_substring=old_substring)
        )

    def remove(self, target: MemoryTarget, substring: str) -> MemoryOpResult:
        return self._apply_single(
            MemoryOp(action="remove", target=target, old_substring=substring)
        )

    def apply_batch(self, ops: list[MemoryOp]) -> MemoryOpResult:
        """批量原子操作。按最终预算校验，任一失败则不写入磁盘。"""
        if not ops:
            return MemoryOpResult(success=True, target=None)

        target_groups: dict[MemoryTarget, list[str]] = {}
        for op in ops:
            if op.target not in target_groups:
                target_groups[op.target] = self.list_entries(op.target)

        last_target: MemoryTarget | None = None
        for op in ops:
            entries = target_groups[op.target]
            result = self._apply_op_to_entries(op, entries)
            if not result.success:
                return result
            last_target = op.target

        for target, entries in target_groups.items():
            limit = self._limit_of(target)
            used = self._calc_usage(entries)
            if used > limit:
                return MemoryOpResult(
                    success=False,
                    target=last_target,
                    error=f"budget exceeded for {target.value}: {used}/{limit} chars",
                    current_entries=target_groups.get(last_target, []),
                    usage={"used": used, "limit": limit},
                )

        try:
            for target, entries in target_groups.items():
                self._write_entries(target, entries)
        except Exception as e:
            return MemoryOpResult(success=False, target=last_target, error=f"write failed: {e}")

        assert last_target is not None
        final_entries = target_groups[last_target]
        return MemoryOpResult(
            success=True,
            target=last_target,
            current_entries=list(final_entries),
            usage={
                "used": self._calc_usage(final_entries),
                "limit": self._limit_of(last_target),
            },
        )

    # ── 内部：单操作实现 ────────────────────────────────

    def _apply_single(self, op: MemoryOp) -> MemoryOpResult:
        """应用单个操作（含威胁扫描、预算校验、磁盘写入）。"""
        if op.action in ("add", "replace") and op.content:
            if scan_strict(op.content):
                return MemoryOpResult(
                    success=False,
                    target=op.target,
                    error="content blocked by threat scan",
                    current_entries=self.list_entries(op.target),
                    usage=self._usage_dict(op.target),
                )

        entries = self.list_entries(op.target)
        result = self._apply_op_to_entries(op, entries)
        if not result.success:
            return result

        limit = self._limit_of(op.target)
        used = self._calc_usage(entries)
        if used > limit:
            return MemoryOpResult(
                success=False,
                target=op.target,
                error=f"limit exceeded: {used}/{limit} chars",
                current_entries=entries,
                usage={"used": used, "limit": limit},
            )

        try:
            self._write_entries(op.target, entries)
        except Exception as e:
            return MemoryOpResult(success=False, target=op.target, error=f"write failed: {e}")

        return MemoryOpResult(
            success=True,
            target=op.target,
            current_entries=list(entries),
            usage={"used": used, "limit": limit},
        )

    def _apply_op_to_entries(self, op: MemoryOp, entries: list[str]) -> MemoryOpResult:
        if op.target == MemoryTarget.USER and not self.user_enabled:
            return MemoryOpResult(success=False, target=op.target, error="user profile disabled")
        """在内存中的 entries 列表上应用操作（不写磁盘）。"""
        if op.action == "add":
            if not op.content:
                return MemoryOpResult(success=False, target=op.target, error="empty content for add")
            if op.content in entries:
                return MemoryOpResult(
                    success=False, target=op.target,
                    error="DUPLICATE - 此内容已存在。你已保存过此信息，不要重复调用 memory，立即回复用户。",
                    current_entries=list(entries),
                )
            entries.append(op.content)
            return MemoryOpResult(success=True, target=op.target, current_entries=list(entries))

        if op.action == "replace":
            if not op.old_substring:
                return MemoryOpResult(success=False, target=op.target, error="missing old_substring for replace")
            matches = [i for i, e in enumerate(entries) if op.old_substring in e]
            if not matches:
                return MemoryOpResult(
                    success=False, target=op.target,
                    error=f"substring not found: '{op.old_substring}'",
                    current_entries=list(entries),
                )
            if len(matches) > 1:
                return MemoryOpResult(
                    success=False, target=op.target,
                    error=f"ambiguous substring '{op.old_substring}' matches {len(matches)} entries",
                    current_entries=list(entries),
                )
            entries[matches[0]] = op.content
            return MemoryOpResult(success=True, target=op.target, current_entries=list(entries))

        if op.action == "remove":
            if not op.old_substring:
                return MemoryOpResult(success=False, target=op.target, error="missing substring for remove")
            matches = [i for i, e in enumerate(entries) if op.old_substring in e]
            if not matches:
                return MemoryOpResult(
                    success=False, target=op.target,
                    error=f"substring not found: '{op.old_substring}'",
                    current_entries=list(entries),
                )
            if len(matches) > 1:
                return MemoryOpResult(
                    success=False, target=op.target,
                    error=f"ambiguous substring '{op.old_substring}' matches {len(matches)} entries",
                    current_entries=list(entries),
                )
            del entries[matches[0]]
            return MemoryOpResult(success=True, target=op.target, current_entries=list(entries))

        return MemoryOpResult(success=False, target=op.target, error=f"unknown action: {op.action}")

    # ── 内部：文件 I/O ────────────────────────────────

    def _path_of(self, target: MemoryTarget) -> Path:
        return self._memory_path if target == MemoryTarget.MEMORY else self._user_path

    def _limit_of(self, target: MemoryTarget) -> int:
        return self.memory_limit if target == MemoryTarget.MEMORY else self.user_limit

    def _usage_dict(self, target: MemoryTarget) -> dict[str, int]:
        entries = self.list_entries(target)
        return {"used": self._calc_usage(entries), "limit": self._limit_of(target)}

    def _calc_usage(self, entries: list[str]) -> int:
        """计算条目列表的字符总用量（含分隔符）。"""
        if not entries:
            return 0
        return sum(len(e) for e in entries) + len(ENTRY_SEPARATOR) * (len(entries) - 1)

    def _read_and_dedup(self, path: Path) -> list[str]:
        """读取文件并按首次出现去重。"""
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        if not text:
            return []
        parts = text.split(ENTRY_SEPARATOR)
        seen: set[str] = set()
        result: list[str] = []
        for p in parts:
            p = p.strip("\n")
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            result.append(p)
        return result

    def _write_entries(self, target: MemoryTarget, entries: list[str]) -> None:
        content = ENTRY_SEPARATOR.join(entries)
        self._write_raw(self._path_of(target), content)

    def _write_raw(self, path: Path, content: str) -> None:
        """原子写入：文件锁 + 临时文件 + os.replace。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with _FileLock(path):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)

    @staticmethod
    def _maybe_block(entry: str) -> str:
        matches = scan_strict(entry)
        if not matches:
            return entry
        names = ", ".join(sorted({m.pattern_name for m in matches}))
        return f"[BLOCKED: {names}]"

    @staticmethod
    def _format_snapshot_text(
        memory_entries: list[str],
        user_entries: list[str],
        memory_usage: tuple[int, int],
        user_usage: tuple[int, int],
    ) -> str:
        parts: list[str] = []

        if memory_entries:
            used, limit = memory_usage
            pct = int(used / limit * 100) if limit > 0 else 0
            parts.append(f"## MEMORY (your personal notes) [{pct}% — {used}/{limit} chars]")
            parts.append("=" * 50)
            parts.extend(memory_entries)
            parts.append("")

        if user_entries:
            used, limit = user_usage
            pct = int(used / limit * 100) if limit > 0 else 0
            parts.append(f"## USER PROFILE (who the user is) [{pct}% — {used}/{limit} chars]")
            parts.append("=" * 50)
            parts.extend(user_entries)
            parts.append("")

        return "\n".join(parts).rstrip()


class _FileLock:
    """跨平台文件锁。"""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(lock_path, "a+", encoding="utf-8")

        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            except OSError:
                pass
        else:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except (OSError, ImportError):
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                try:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                except (OSError, ImportError):
                    pass
        finally:
            self._fh.close()
            self._fh = None