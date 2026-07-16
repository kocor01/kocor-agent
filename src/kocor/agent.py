"""Agent 核心。

Agent 身份、slash 命令路由、组件组装。ReAct 循环由 Loop 引擎驱动。
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
from kocor.memory.store import MemoryStore

# 可选的会话管理
from kocor.session.manager import SessionManager as _SessionManager
from kocor.skill.types import InvokeStrategy, SkillContext, SkillType
from kocor.tools.permission import PermissionManager
from kocor.tools.tool_manager import ToolManager

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
        tool_manager: ToolManager | None = None,
        # Harness 参数（可选）
        permission_mgr: PermissionManager | None = None,
        hook_manager: HookManager | None = None,
        event_emitter: EventEmitter | None = None,
        max_iterations: int | None = None,
        # 会话管理（可选）
        session_manager: _SessionManager | None = None,
    ):
        self.llm = llm
        self.tool_manager = tool_manager or ToolManager()
        self.system_prompt = Config.load().default_system_prompt
        # Harness 组件
        self.permission_mgr = permission_mgr or PermissionManager(policy=PermissionManager.POLICY_PERMISSIVE)
        self.hook_manager = hook_manager or HookManager()
        self.event_emitter = event_emitter or EventEmitter()
        self.max_iterations = max_iterations or Config.load().max_iterations

        # 会话管理
        self.session_manager = session_manager
        self._persisted_msg_idx = 0

        # Cron 调度器标志（首次使用时启动）
        self._cron_started = False

        # 上下文管理
        self._memory: MemoryStore | None = None
        self._background_reviewer = None
        self._turns_since_memory = 0
        if Config.load().memory_enabled:
            memory_dir = Config.load().memory_dir or None
            if memory_dir:
                self._memory = MemoryStore(
                    memory_dir=memory_dir,
                    memory_limit=Config.load().memory_char_limit,
                    user_limit=Config.load().user_char_limit,
                    user_enabled=Config.load().user_profile_enabled,
                )
                self._memory.load_from_disk()
                self.tool_manager.memory_store = self._memory
                from kocor.memory.reviewer import BackgroundReviewer
                self._background_reviewer = BackgroundReviewer(llm=self.llm, store=self._memory)

        # 任务规划（todo）：零依赖、零风险，始终启用
        from kocor.tools.toolsets.todo_tool import TodoStore
        self._todo_store = TodoStore()
        self.tool_manager.todo_store = self._todo_store

        # 运行时上下文管理器
        self.ctx = ContextManager(
            tools=self.tool_manager,
            memory=self._memory,
            todo_store=self._todo_store,
        )

        # ReAct 循环引擎：Agent 负责组装并拥有 harness 组件，Loop 仅持有引用、
        # 专注迭代机制。调用方通过 Loop 公共入口驱动循环，不访问其私有成员。
        self.loop = Loop(
            llm=self.llm,
            ctx=self.ctx,
            tool_manager=self.tool_manager,
            permission_mgr=self.permission_mgr,
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
        self.ctx.reset_conversation()
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
            self.ctx.append(Message(
                role="user",
                content=f"[会话因 {reason} 已自动重置，开始新对话]",
            ))
            self._persisted_msg_idx = 0

        # 恢复历史消息（跨进程重启时 session_history 为空）
        if not self.ctx.session_history and self.session_manager.store.db:
            history = self.session_manager.load_messages(entry.session_id)
            if history:
                self.ctx.session_history = history
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

        # 使用 _persisted_msg_idx 作为已持久化消息的基准，
        # 而非 entry.message_count，因为会话可能在 ReAct 循环
        # 中被重置（如 /reset 或自动重置），此时 entry 指向
        # 新会话且 message_count 为 0，但 session_history 中
        # 可能已有本轮新增消息
        prev_count = self._persisted_msg_idx
        current_total = len(self.ctx.session_history)
        msg_delta = max(0, current_total - prev_count)

        # token 用量从新增消息的 usage 累计（ReAct 循环可能有多次 API 调用）
        prompt_delta = 0
        completion_delta = 0
        total_delta = 0
        cached_delta = 0
        for i in range(prev_count, current_total):
            msg = self.ctx.session_history[i]
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

        # 持久化新增消息
        self._persisted_msg_idx = self.session_manager.persist_messages(
            session_key=session_key,
            messages=self.ctx.session_history,
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
        self.ctx.reset_conversation()
        self._persisted_msg_idx = 0
        self.ctx.session_history = messages
        self._hydrate_todo_store(messages)

        return f"✅ 已切换到会话 {target_id}（{len(messages)} 条消息）。\n你可以继续之前的对话了。"

    # ── slash 命令 ──

    def _handle_slash_command(self, user_input: str) -> str:
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
            messages = [
                Message(role="system", content=self.system_prompt),
            ]
            if skill_def.prompt_role == "system":
                messages.append(Message(role="system", content=result.content))
            else:
                messages.append(Message(role="user", content=result.content))
            self.ctx.reset()
            self.ctx.messages = messages
            # 走 Loop 公共入口：循环结束后由 Loop 统一提取 session_history，
            # 随后 _session_after_run 据此持久化 PROMPT 技能触发的 LLM 回复
            return self.loop.run_messages()
        else:
            return result.content

    def _list_slash_skills(self) -> str:
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
        if not self._background_reviewer:
            return
        self._turns_since_memory += 1
        nudge_interval = Config.load().nudge_interval
        if self._turns_since_memory >= nudge_interval:
            self._background_reviewer.review(self.ctx.session_history)
            # 审查后刷新快照，使新记忆立即对 LLM 可见
            if self._memory:
                self._memory.refresh_snapshot()
            self._turns_since_memory = 0
