"""Agent 核心。

Agent 身份、slash 命令路由。组件装配由 AgentBuilder 负责，ReAct 循环由 Loop 引擎驱动。
"""

from __future__ import annotations

import logging
from typing import Iterator

from kocor.config import Config
from kocor.context.context_manager import ContextManager
from kocor.event.event_manager import EventEmitter
from kocor.hook.hook_manager import HookManager
from kocor.llm_provider.llm_client import LLMClient
from kocor.llm_provider.message import Message, StreamChunk
from kocor.loop import Loop
from kocor.memory.reviewer import BackgroundReviewer
from kocor.memory.store import MemoryStore

# 可选的会话管理
from kocor.session.manager import SessionManager as _SessionManager
from kocor.skill.types import InvokeStrategy, SkillContext, SkillType
from kocor.tools.tool_manager import ToolManager
from kocor.tools.toolsets.todo_tool import TodoStore

logger = logging.getLogger(__name__)


class Agent:
    """自主 Agent 核心。

    职责：
    - slash 命令识别和调度
    - Agent 身份管理（LLM、工具、上下文）
    - 委托 Loop 执行 ReAct 循环
    - 可选的会话管理（SessionManager）
    """

    def __init__(
        self,
        llm: LLMClient,
        tool_manager: ToolManager,  # 必传
        todo_store: TodoStore,  # 必传
        context: ContextManager,  # 必传
        hook_manager: HookManager,  # 必传
        event_emitter: EventEmitter,  # 必传
        max_iterations: int,  # 必传
        # 会话管理（可选）
        session_manager: _SessionManager | None = None,
        # 记忆/审查（可选，通过 if 守卫安全使用）
        memory: MemoryStore | None = None,
        background_reviewer: BackgroundReviewer | None = None,
    ):
        """初始化 Agent。

        Args:
            llm: LLM 客户端
            tool_manager: 工具管理器
            todo_store: 任务列表存储
            context: 运行时上下文管理器
            hook_manager: 钩子管理器
            event_emitter: 事件发射器
            max_iterations: 最大 ReAct 迭代次数
            session_manager: 会话管理器（None 为无会话持久化）
            memory: 记忆存储（None 为无记忆）
            background_reviewer: 后台记忆审查器（None 为无审查）
        """
        self.llm = llm
        self.tool_manager = tool_manager
        self.system_prompt = Config.load().default_system_prompt
        self.hook_manager = hook_manager
        self.event_emitter = event_emitter
        self.max_iterations = max_iterations

        # 会话管理
        self.session_manager = session_manager
        # 已持久化的消息索引——标记 session_history 中被写入 SQLite 的最后位置。
        # 用于自上次 persist 后增量写入，避免每次全量转储。
        self._persisted_msg_idx = 0

        # Cron 调度器标志（首次使用时启动）
        self._cron_started = False

        # 运行时指标收集器（由 AgentBuilder 在 build() 时通过 setattr 注入，初始为 None）
        self._metrics_collector = None

        # 上下文管理：记忆系统、后台审查、轮次计数器
        self._memory: MemoryStore | None = memory
        self._background_reviewer = background_reviewer
        self._turns_since_memory = 0

        # 任务列表
        self._todo_store = todo_store
        self.tool_manager.todo_store = self._todo_store  # 共享给 todo 工具

        # 运行时上下文
        self.context = context

        # ReAct 循环引擎：Agent 负责组装并拥有 harness 组件，Loop 仅持有引用、
        # 专注迭代机制。调用方通过 Loop 公共入口驱动循环，不访问其私有成员。
        self.loop = Loop(
            llm=self.llm,
            context=self.context,
            tool_manager=self.tool_manager,
            hook_manager=self.hook_manager,
            event_emitter=self.event_emitter,
            max_iterations=self.max_iterations,
        )

    # ── 公开方法 ──

    def _execute_with_session(self, user_input: str) -> str | None:
        """执行会话管理前置/后置工作。

        返回 None 表示继续执行 ReAct 循环；
        返回非 None 表示命令已处理完毕（如内置命令、slash 命令），可直接返回结果。
        """
        self._ensure_cron_started()
        if user_input.startswith("/"):
            cmd_result = self._handle_builtin_commands(user_input)
            if cmd_result is not None:
                return cmd_result
        self._session_before_run()
        if self.tool_manager.skill_manager and user_input.startswith("/"):
            result = self._handle_slash_command(user_input)
            self._session_after_run()
            self._check_nudge()
            return result
        return None  # 继续执行 ReAct 循环

    def stop(self) -> None:
        """请求在当前迭代边界停止 ReAct 循环。"""
        self.tool_manager.stop_cron_scheduler()
        self._cron_started = False
        self.loop.stop()

    def metrics(self) -> dict | None:
        """返回当前会话的运行时指标摘要（若有收集器）。"""
        collector = getattr(self, "_metrics_collector", None)
        if collector is None:
            return None
        return collector.report()

    def run(self, user_input: str) -> str:
        """执行一次完整的 Agent 循环。"""
        result = self._execute_with_session(user_input)
        if result is not None:
            return result
        result = self.loop.run(user_input)
        self._session_after_run()
        self._check_nudge()
        return result

    def run_prompt(self, prompt: str, skills: list[str] | None = None) -> str:
        """为 cron 作业执行一次隔离的 ReAct 循环（不启动 cron worker）。

        cron worker 子进程内由 CronScheduler._execute_job 调用。
        每次返回独立的最终输出文本——不启动 cron worker（防递归），
        不使用会话管理，不检查 nudge。

        Args:
            prompt: 用户提示词
            skills: 技能列表（预留，当前仅记录日志）

        Returns:
            LLM 最终输出文本
        """
        if skills:
            logger.debug("run_prompt skills not yet implemented: %s", skills)
        # 直接运行 ReAct 循环：loop.run 重置上下文 → build_initial_context → run_messages
        return self.loop.run(prompt)

    def _ensure_cron_started(self) -> None:
        """确保 cron worker 子进程已启动（首次 run 时，或崩溃后重新启动）。"""
        worker = getattr(self.tool_manager, "cron_worker", None)
        if worker is not None and not worker.is_running:
            logger.info("cron worker 子进程未运行，重新启动")
            
            self.tool_manager.start_cron_scheduler()
            self._cron_started = True
        elif not self._cron_started:
            self.tool_manager.start_cron_scheduler()
            self._cron_started = True

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 Agent 循环。"""
        result = self._execute_with_session(user_input)
        if result is not None:
            yield StreamChunk(content=result, is_final=True)
            return
        yield from self.loop.stream(user_input)
        self._session_after_run()
        self._check_nudge()

    def reset_conversation(self) -> None:
        """清空会话历史，开始新对话。

        重置时刷新记忆快照，确保新对话看到最新记忆。
        """
        self.context.reset_conversation()
        self._persisted_msg_idx = 0
        if self._memory:
            self._memory.refresh_snapshot()
        if self.session_manager:
            self.session_manager.reset_session(self.session_manager.session_key)

    # ── 会话管理 ──

    def _session_before_run(self) -> None:
        """执行一次 ReAct 循环前的会话准备工作。

        1. 获取/创建会话
        2. 自动重置时注入通知
        3. 从 SQLite 恢复历史消息（如跨进程重启）
        """
        if not self.session_manager:
            return

        entry = self.session_manager.get_or_create_session()

        # 自动重置时注入通知，重置已持久化消息索引
        if entry.was_auto_reset:
            reason = entry.auto_reset_reason or "unknown"
            # 向 LLM 注入一条系统通知，说明会话已重置，让其感知上下文变化
            self.context.append(Message(
                role="user",
                content=f"[会话因 {reason} 已自动重置，开始新对话]",
            ))
            self._persisted_msg_idx = 0

        # 恢复历史消息（跨进程重启时 session_history 为空）
        if not self.context.session_history and self.session_manager.store.db:
            history = self.session_manager.load_messages(entry.session_id)
            if history:
                self.context.session_history = history
                self._persisted_msg_idx = len(history)
                self._hydrate_todo_store(history)

    def _session_after_run(self) -> None:
        """一次 ReAct 循环后的会话收尾工作。

        1. 更新会话元数据（消息数、token 数）
        2. 持久化新增消息到 SQLite
        """
        if not self.session_manager:
            return

        session_key = self.session_manager.session_key
        entry = self.session_manager.get_session_info(session_key)
        if entry is None:
            return

        # 计算本轮新增消息数：用 _persisted_msg_idx 而非 entry.message_count，
        # 因为会话可能在 ReAct 循环中被重置（如 /reset），此时 entry 指向新会话
        # 且 message_count=0，但 session_history 中可能有本轮新增消息。
        # 使用本地索引避免了 entry 状态与真实数据不同步的问题。
        prev_count = self._persisted_msg_idx
        current_total = len(self.context.session_history)
        msg_delta = max(0, current_total - prev_count)

        # token 用量从新增消息的 usage 累计（ReAct 循环可能有多次 API 调用，
        # 每次 API 调用返回的 usage 附着在对应 assistant 消息上）
        prompt_delta = 0
        completion_delta = 0
        total_delta = 0
        cached_delta = 0
        for i in range(prev_count, current_total):
            msg = self.context.session_history[i]
            if msg.usage:
                prompt_delta += msg.usage.prompt_tokens
                completion_delta += msg.usage.completion_tokens
                total_delta += msg.usage.total_tokens
                cached_delta += msg.usage.cached_tokens

        self.session_manager.update_session(
            session_key=session_key,
            message_count_delta=msg_delta,
            prompt_tokens_delta=prompt_delta,
            completion_tokens_delta=completion_delta,
            total_tokens_delta=total_delta,
            cached_tokens_delta=cached_delta,
        )

        # 持久化新增消息（增量写入，避免每次全量转储）
        self._persisted_msg_idx = self.session_manager.persist_messages(
            session_key=session_key,
            messages=self.context.session_history,
            start_index=prev_count,
        )

    # ── 内置命令 ──

    def _handle_builtin_commands(self, user_input: str) -> str | None:
        """处理内置 slash 命令（会话管理相关）。返回 None 表示非内置命令。"""
        parts = user_input[1:].strip().split(maxsplit=1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("reset", "new"):
            self.reset_conversation()
            return "✅ 会话已重置。" if cmd == "reset" else "✅ 已创建新会话。"
        if cmd == "sessions" and self.session_manager:
            return self._handle_list_sessions()
        if cmd == "session" and self.session_manager:
            return self._handle_switch_session(args)
        return None  # 非内置命令，继续走 skill 路由

    @staticmethod
    def _format_sessions_table(sessions: list[dict]) -> str:
        """将会话列表格式化为表格文本。"""
        if not sessions:
            return "📋 暂无历史会话。\n\n使用 `KOCOR_SESSION_ENABLED=1` 启用会话持久化后自动记录。"

        lines = [
            "📋 历史会话",
            f"{'#':>3}  {'Session ID':<24}  {'创建时间':<14}  {'消息':>4}  摘要",
            f"{'─'*3}  {'─'*24}  {'─'*14}  {'─'*4}  {'─'*20}",
        ]
        for i, s in enumerate(sessions, 1):
            sid = s["session_id"]
            created = s["created_at"][:16].replace("T", " ")
            n = s["message_count"]
            preview = s["title"] or "(空)"
            lines.append(f"{i:>3}  {sid:<24}  {created:<14}  {n:>4}  {preview}")

        lines.append("")
        lines.append("使用 `/session <序号或ID>` 重新进入某个会话。")
        return "\n".join(lines)

    def _handle_list_sessions(self) -> str:
        """处理 /sessions 命令。"""
        if not self.session_manager:
            return "⚠️ 会话管理未启用。"
        sessions = self.session_manager.get_sessions_list()
        return self._format_sessions_table(sessions)

    def _handle_switch_session(self, args: str) -> str:
        """处理 /session <id|序号> 命令。"""
        if not self.session_manager:
            return "⚠️ 会话管理未启用。"

        if not args.strip():
            return self._handle_list_sessions()

        sessions = self.session_manager.get_sessions_list()
        if not sessions:
            return "⚠️ 无可切换的历史会话。"

        target_id = args.strip()

        # 按序号查找
        if target_id.isdigit():
            idx = int(target_id) - 1
            if 0 <= idx < len(sessions):
                target_id = sessions[idx]["session_id"]
            else:
                return f"⚠️ 序号 {target_id} 超出范围（共 {len(sessions)} 个会话）。"

        # 按 session_id 查找
        elif not any(s["session_id"] == target_id for s in sessions):
            return f"⚠️ 未找到会话: {target_id}"

        # 切换会话
        session_key = self.session_manager.session_key
        messages = self.session_manager.switch_to_session(
            session_key=session_key,
            target_session_id=target_id,
        )
        if messages is None:
            return f"⚠️ 无法切换到会话 {target_id}。"

        # 恢复上下文
        self.context.reset_conversation()
        self._persisted_msg_idx = 0
        self.context.session_history = messages
        self._hydrate_todo_store(messages)

        return f"✅ 已切换到会话 {target_id}（{len(messages)} 条消息）。\n你可以继续之前的对话了。"

    # ── slash 命令 ──

    def _handle_slash_command(self, user_input: str) -> str:
        """解析 /<skill_name> [args] 格式并执行技能。

        根据技能类型调度:
          - PROMPT 技能: 将渲染结果注入 LLM 上下文，走 ReAct 循环
          - CODE 技能: 直接返回执行结果，不走 LLM
        """
        parts = user_input[1:].strip().split(maxsplit=1)
        skill_name = parts[0]
        skill_args = parts[1] if len(parts) > 1 else ""

        skill_def = self.tool_manager.skill_manager.get(skill_name)
        if skill_def is None:
            available = self._list_slash_skills()
            return f"Unknown skill: '{skill_name}'. Available: {available}"

        if skill_def.invoke_strategy not in (InvokeStrategy.SLASH, InvokeStrategy.BOTH):
            return f"Skill '{skill_name}' cannot be invoked via slash command."

        context = SkillContext(
            user_input=skill_args,
            tool_manager=self.tool_manager,
        )
        result = self.tool_manager.skill_manager.execute(skill_name, context)

        if not result.success:
            return result.content

        if skill_def.skill_type == SkillType.PROMPT:
            # PROMPT 技能：渲染后的 prompt 走 ReAct 循环，让 LLM 处理后返回
            messages = [
                Message(role="system", content=self.system_prompt),
            ]
            if skill_def.prompt_role == "system":
                # 技能注入为 system 消息（如设定角色行为）
                messages.append(Message(role="system", content=result.content))
            else:
                # 技能注入为 user 消息（默认）
                messages.append(Message(role="user", content=result.content))
            self.context.reset()
            self.context.messages = messages
            # 走 Loop 公共入口：循环结束后由 Loop 统一提取 session_history，
            # 随后 _session_after_run 据此持久化 PROMPT 技能触发的 LLM 回复
            return self.loop.run_messages()
        else:
            # CODE 技能：直接返回执行结果，不走 LLM 循环
            return result.content

    def _list_slash_skills(self) -> str:
        """返回所有支持斜杠调用的技能名称列表（逗号分隔）。"""
        names = [
            f"/{s.name}"
            for s in self.tool_manager.skill_manager.list_skills(enabled_only=True)
            if s.invoke_strategy in (InvokeStrategy.SLASH, InvokeStrategy.BOTH)
        ]
        return ", ".join(sorted(names))

    # ── 记忆审查 ──

    def _hydrate_todo_store(self, history) -> None:
        """从历史消息回填 TodoStore（仅在 store 为空时，避免覆盖实时状态）。"""
        if not self._todo_store.has_items():
            self._todo_store.hydrate_from_history(history)

    def _check_nudge(self) -> None:
        """检查是否需要触发后台记忆审查。

        审查后刷新记忆快照，使新写入的记忆对下一轮 LLM 调用可见。
        """
        if not self._memory or not self._background_reviewer:
            return
        self._turns_since_memory += 1
        nudge_interval = Config.load().nudge_interval
        if self._turns_since_memory >= nudge_interval:
            self._background_reviewer.review(self.context.session_history)
            # 审查后刷新快照，使新记忆立即对 LLM 可见
            self._memory.refresh_snapshot()
            self._turns_since_memory = 0
