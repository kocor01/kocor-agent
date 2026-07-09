"""Cron 任务工具的类型定义与常量。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 默认 tick 轮询间隔（秒）
DEFAULT_TICK_INTERVAL = 60

# 一次性作业的 grace window（秒）
ONESHOT_GRACE_SECONDS = 120

# 运行锁 TTL 下限（秒）
ONESHOT_RUN_CLAIM_TTL_SECONDS = 1800

# 每个作业保留的最大输出文件数
CRON_OUTPUT_DEFAULT_KEEP = 50

# Cron 作业存储相关路径
CRON_DIR_NAME = "cron"
JOBS_FILE_NAME = "jobs.json"
OUTPUT_DIR_NAME = "output"
TICKER_LOCK_NAME = ".tick.lock"
JOBS_LOCK_NAME = ".jobs.lock"
TICKER_HEARTBEAT_NAME = "ticker_heartbeat"

# Cron agent 会话中禁用的工具集
DISABLED_TOOLSETS_IN_CRON = frozenset({"cronjob"})

# 作业合法状态值
STATE_SCHEDULED = "scheduled"
STATE_PAUSED = "paused"
STATE_COMPLETED = "completed"
STATE_ERROR = "error"

# 状态列表
VALID_STATES = {STATE_SCHEDULED, STATE_PAUSED, STATE_COMPLETED, STATE_ERROR}


@dataclass
class ScheduleInfo:
    """解析后的调度信息。"""

    kind: str  # once | interval | cron
    display: str = ""
    minutes: int | None = None  # interval
    expr: str | None = None  # cron
    run_at: str | None = None  # once


@dataclass
class CronJob:
    """Cron 作业数据模型。"""

    id: str
    name: str
    prompt: str
    schedule: dict[str, Any]
    schedule_display: str
    enabled: bool = True
    state: str = STATE_SCHEDULED
    created_at: str = ""
    next_run_at: str | None = None
    last_run_at: str | None = None
    last_status: str | None = None  # ok | error
    last_error: str | None = None

    # 重复设置
    repeat: dict | None = None  # {"times": N|None, "completed": N}

    # 脚本模式
    script: str | None = None
    no_agent: bool = False

    # 链式依赖
    context_from: list[str] | None = None

    # 工具集限制
    enabled_toolsets: list[str] | None = None
    workdir: str | None = None

    # 暂停
    paused_at: str | None = None
    paused_reason: str | None = None

    # 执行锁（at-most-once CAS）
    fire_claim: dict | None = None
    run_claim: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        result: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is not None:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CronJob:
        """从字典创建实例。"""
        field_names = {f.name for f in field(cls)}
        kwargs = {k: v for k, v in data.items() if k in field_names}
        return cls(**kwargs)

    @property
    def is_recurring(self) -> bool:
        """是否为循环作业。"""
        kind = self.schedule.get("kind", "")
        return kind in ("interval", "cron")

    @property
    def is_one_shot(self) -> bool:
        """是否为一次性作业。"""
        return self.schedule.get("kind") == "once"

    @property
    def repeat_times(self) -> int | None:
        """获取重复次数（None=无限）。"""
        if self.repeat:
            return self.repeat.get("times")
        return None

    @property
    def repeat_completed(self) -> int:
        """获取已完成次数。"""
        if self.repeat:
            return self.repeat.get("completed", 0)
        return 0