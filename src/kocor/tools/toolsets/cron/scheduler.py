"""Cron 作业执行引擎 —— 后台 tick 线程调度器。

职责：
- 启动后台线程定期轮询到期作业
- 抢占式 at-most-once 执行保障
- 输出保存
- 生命周期管理（启动/停止）
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from kocor.tools.toolsets.cron.jobs import (
    claim_job_for_fire,
    get_due_jobs,
    mark_job_run,
    save_job_output,
)
from kocor.tools.toolsets.cron.types import DEFAULT_TICK_INTERVAL

logger = logging.getLogger(__name__)


class CronScheduler:
    """定时任务调度器。

    通过后台线程定期检查到期作业并执行。
    支持安全的启动/停止生命周期管理。
    """

    def __init__(
        self,
        tick_interval: int = DEFAULT_TICK_INTERVAL,
        agent: Any = None,
    ):
        """
        Args:
            tick_interval: tick 轮询间隔（秒），默认 60s
            agent: 子进程内的独立 Agent，用于执行 prompt 作业。
                仅 cron worker 子进程注入；主进程不构造 CronScheduler。
                为 None 时 prompt 作业退化为占位输出（兼容无 LLM / 测试场景）。
        """
        self._tick_interval = tick_interval
        self._tick_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._agent = agent

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """调度器是否正在运行。"""
        return self._running

    @property
    def agent(self) -> Any:
        """注入的独立 Agent（cron worker 子进程内）。"""
        return self._agent

    def start(self) -> None:
        """启动后台 tick 线程（守护线程，不会阻止进程退出）。"""
        if self._running:
            logger.warning("CronScheduler 已在运行")
            return

        self._stop_event.clear()
        self._running = True
        self._tick_thread = threading.Thread(
            target=self._tick_loop,
            name="cron-ticker",
            daemon=True,
        )
        self._tick_thread.start()
        logger.info(
            "CronScheduler 已启动 (interval=%ds, daemon=True)",
            self._tick_interval,
        )

    def stop(self) -> None:
        """停止后台 tick 线程。"""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False
        if self._tick_thread and self._tick_thread is not threading.current_thread():
            self._tick_thread.join(timeout=5)
        self._tick_thread = None
        logger.info("CronScheduler 已停止")

    # ------------------------------------------------------------------
    # Tick 循环
    # ------------------------------------------------------------------

    def _tick_loop(self) -> None:
        """后台轮询循环。"""
        logger.debug("CronScheduler tick 循环开始")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("CronScheduler tick 发生异常，继续下一轮")

            # 等间隔等待（支持提前停止）
            self._stop_event.wait(self._tick_interval)

        logger.debug("CronScheduler tick 循环结束")

    def _tick(self) -> None:
        """单次 tick：获取到期作业并执行。"""
        due_jobs = get_due_jobs()
        if not due_jobs:
            return

        logger.debug("CronScheduler tick: %d 个到期作业", len(due_jobs))

        for job in due_jobs:
            job_id = job["id"]
            job_name = job.get("name", job_id)

            # CAS 抢占执行权
            if not claim_job_for_fire(job_id):
                logger.debug("作业 %s 已被其他线程抢占，跳过", job_name)
                continue

            try:
                self._execute_job(job)
                mark_job_run(job_id, success=True)
                logger.info("作业 %s 执行成功", job_name)
            except Exception as e:
                logger.exception("作业 %s 执行失败: %s", job_name, e)
                try:
                    mark_job_run(job_id, success=False, error=str(e))
                except Exception:
                    logger.exception("标记作业 %s 执行状态失败", job_name)

    # ------------------------------------------------------------------
    # 作业执行
    # ------------------------------------------------------------------

    def _execute_job(self, job: dict[str, Any]) -> str:
        """执行单个作业。

        Args:
            job: 作业字典

        Returns:
            执行输出字符串

        Raises:
            Exception: 执行过程中的任何异常
        """
        prompt = job.get("prompt", "") or ""
        script = job.get("script")
        no_agent = bool(job.get("no_agent", False))

        # 构建输出
        output_parts: list[str] = []

        if no_agent and script:
            # no_agent 模式：执行脚本
            output = self._run_script(script, job)
            output_parts.append(output)
        elif prompt:
            # prompt 模式：委托子进程内独立 Agent 运行 ReAct 循环。
            # 无 agent（测试 / 无 LLM）时退化为占位输出。
            if self._agent is not None:
                output_parts.append(
                    self._agent.run_prompt(prompt, job.get("skills") or [])
                )
            else:
                output_parts.append(f"[Cron job execution]\nPrompt: {prompt}\n")
                output_parts.append(
                    "(No agent configured — LLM execution unavailable.)"
                )

        # 添加技能信息
        skills = job.get("skills", [])
        if skills:
            output_parts.append(f"Skills: {', '.join(skills)}")

        output = "\n".join(output_parts)

        # 保存输出到文件
        try:
            save_job_output(job["id"], output)
        except Exception as e:
            logger.warning("保存作业 %s 输出失败: %s", job.get("name", job["id"]), e)

        return output

    def _run_script(self, script: str, job: dict[str, Any]) -> str:
        """执行脚本。"""
        import subprocess

        try:
            result = subprocess.run(
                script,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                raise RuntimeError(
                    f"Script exited with code {result.returncode}: {error_msg}"
                )
            return output.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Script timed out after 300s: {script}")
        except FileNotFoundError:
            raise RuntimeError(f"Script not found: {script}")