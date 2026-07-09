"""Cron 任务工具的类型定义与常量。"""

from __future__ import annotations

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