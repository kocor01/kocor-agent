"""Kocor Agent CLI 入口。

使用:
    python -m kocor "你的问题"
    python -m kocor --stream "你的问题"
    python -m kocor --repl           # 交互模式
    echo "你的问题" | python -m kocor
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from io import StringIO
from typing import Any, Iterator

from rich.console import Console
from rich.markdown import Markdown

from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_manager import LlmManager
from kocor.llm_provider.message import StreamChunk
from kocor.skill.types import InvokeStrategy
from kocor.tools.tool_manager import ToolManager

# Harness imports
from kocor.harness import IterationBudget
from kocor.tools.permission import PermissionManager
from kocor.hook.hook_manager import HookManager
from kocor.harness.event.event_manager import EventEmitter
from kocor.harness.event.event_subscribe import EventSubscribe
from kocor.harness.logger import setup_logger

# 可选的会话管理
from kocor.session import SessionManager, SessionResetPolicy, SessionStore

W = 58


class _StreamFormatter:
    """管理流式输出的格式状态。"""

    def __init__(self, width: int = W):
        self.width = width
        self.round_num = 0
        self.pending_round = False
        self.tool_calls: list = []
        self.has_reasoning = False
        self.has_content = False
        self.has_tool_section = False
        self.tool_result_idx = 0
        self._content_has_printed_any = False
        self._content_buffer: str = ""
        self._in_code_block: bool = False
        self._code_block_buffer: str = ""
        self._block_buffer: str = ""

    def _round_header(self, n: int) -> None:
        title = f"⚡ 第 {n} 次请求"
        fill = self.width - 2 - len(title)
        print(f"\n── {title} {'─' * max(0, fill)}")

    def _render_markdown(self, text: str) -> None:
        """将已完整的段落文本通过 print() 输出（内部用 rich Markdown 渲染）。"""
        text = text.strip()
        if not text:
            return
        # 渲染到 StringIO，再通过 print() 输出（保持与 mock-print 测试兼容）
        buf = StringIO()
        Console(file=buf, width=self.width).print(Markdown(text))
        rendered = buf.getvalue()
        if rendered:
            print(rendered, end="", flush=True)

    def _flush_block(self) -> None:
        """刷新并渲染已累积的文本块（表格、段落等）。"""
        block = self._block_buffer.strip()
        self._block_buffer = ""
        if block:
            self._render_markdown(block)

    def handle_chunk(self, chunk) -> None:
        if not chunk.tool_result and not self._in_round():
            self.round_num += 1
            self.pending_round = True

        if self.pending_round and (chunk.reasoning or chunk.content or chunk.tool_calls or chunk.is_final):
            self._round_header(self.round_num)
            self.pending_round = False

        self._handle_reasoning(chunk)
        self._handle_content(chunk)
        self._handle_tool_calls(chunk)
        self._handle_tool_results(chunk)
        self._handle_end_of_round(chunk)

    def _in_round(self) -> bool:
        return bool(self.pending_round or self.has_reasoning or self.has_content or self.has_tool_section)

    def _handle_reasoning(self, chunk) -> None:
        if not chunk.reasoning:
            return
        if not self.has_reasoning:
            print("\n\U0001f9e0 思维过程")
            print(f"{'─' * self.width}")
            self.has_reasoning = True
        print(chunk.reasoning, end="", flush=True)

    def _handle_content(self, chunk) -> None:
        if not chunk.content:
            return
        # 去除前导空行（只在首次输出时）
        content = chunk.content
        if not self._content_has_printed_any:
            content = content.lstrip("\n")
        if not content:
            return
        if not self.has_content:
            print("\n\U0001f4ac 回答内容")
            print(f"{'─' * self.width}")
            self.has_content = True
            self._content_has_printed_any = True

        # 累积内容到缓冲区
        self._content_buffer += content

        # 按行拆分处理：累积文本块，按空行边界整体渲染（支持表格等跨行结构）
        while "\n" in self._content_buffer:
            line, self._content_buffer = self._content_buffer.split("\n", 1)
            line = line.rstrip()

            # 检测代码块边界（``` 开头）
            if line.startswith("```"):
                if self._in_code_block:
                    # 代码块闭合前先刷新未闭合的文本块
                    self._flush_block()
                    # 代码块闭合——整块渲染
                    self._code_block_buffer += line
                    self._render_markdown(self._code_block_buffer)
                    self._code_block_buffer = ""
                    self._in_code_block = False
                else:
                    # 代码块开始前先刷新前面的文本块
                    self._flush_block()
                    # 代码块开始——进入批模式
                    self._in_code_block = True
                    self._code_block_buffer = line + "\n"
                continue

            if self._in_code_block:
                self._code_block_buffer += line + "\n"
            elif line == "":
                # 空行 = 块边界，渲染已累积的表格块
                self._flush_block()
            elif re.fullmatch(r"[-*_]{3,}\s*", line):
                # Markdown 水平线（--- / *** / ___）→ 输出简洁分隔线
                self._flush_block()
                print(f"{'─' * max(4, self.width)}")
            elif line.startswith("|"):
                # 表格行——累积到缓冲区，等待整表渲染
                self._block_buffer += line + "\n"
            else:
                # 普通行——先刷新未闭合的表格块，再即时渲染
                self._flush_block()
                self._render_markdown(line)

    def _handle_tool_calls(self, chunk) -> None:
        if not chunk.tool_calls:
            return
        seen_ids = {tc.id for tc in self.tool_calls}
        for tc in chunk.tool_calls:
            if tc.id not in seen_ids:
                self.tool_calls.append(tc)
                seen_ids.add(tc.id)
        if not self.has_tool_section and not chunk.tool_result:
            print("\n\U0001f527 工具调用")
            print(f"{'─' * self.width}")
            self.has_tool_section = True

    def _handle_tool_results(self, chunk) -> None:
        if not chunk.tool_result:
            return
        if self.tool_result_idx < len(self.tool_calls):
            tc = self.tool_calls[self.tool_result_idx]
            content = chunk.tool_result.content
            if len(content) > 1000:
                content = content[:1000] + "..."
            if self.tool_result_idx > 0:
                print()
            print(f"{self.tool_result_idx + 1:2}. {tc.function.name}({tc.function.arguments})")
            print(f"{'─' * (self.width - 4)}")
            print(f"{content}")
            self.tool_result_idx += 1

    def _handle_end_of_round(self, chunk) -> None:
        if not chunk.is_final:
            return
        # 刷新残留缓冲区中的内容
        self._flush_block()
        if self._content_buffer.strip():
            self._render_markdown(self._content_buffer)
        self._content_buffer = ""
        # 未闭合代码块强制刷新
        if self._code_block_buffer:
            self._render_markdown(self._code_block_buffer)
            self._code_block_buffer = ""
            self._in_code_block = False
        self.has_reasoning = False
        self.has_content = False
        self.has_tool_section = False
        self._content_has_printed_any = False
        if chunk.tool_result and self.tool_result_idx >= len(self.tool_calls) and self.tool_calls:
            self.tool_calls.clear()
            self.tool_result_idx = 0

    def flush_remaining(self) -> None:
        self._flush_block()
        if self._content_buffer.strip():
            self._render_markdown(self._content_buffer)
            self._content_buffer = ""
        if self._code_block_buffer:
            self._render_markdown(self._code_block_buffer)
            self._code_block_buffer = ""
            self._in_code_block = False
        for tc in self.tool_calls[self.tool_result_idx:]:
            print(f"• {tc.function.name}({tc.function.arguments})")


def _print_stream_formatted(chunks: Iterator[StreamChunk]) -> None:
    formatter = _StreamFormatter()
    for chunk in chunks:
        formatter.handle_chunk(chunk)
    formatter.flush_remaining()


def parse_args():
    parser = argparse.ArgumentParser(description="Kocor Agent - 小而美的 LLM 自主 Agent 助手")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用流式输出",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="交互式 REPL 模式",
    )
    parser.add_argument(
        "--permissive",
        action="store_true",
        help="允许危险操作（工具调用无需确认）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式（每次工具调用都确认）",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="最大迭代次数",
    )
    parser.add_argument(
        "--session",
        action="store_true",
        default=False,
        help="启用会话持久化（等效于 KOCOR_SESSION_ENABLED=1）",
    )
    parser.add_argument(
        "user_input",
        nargs="*",
        help="用户问题",
    )
    args = parser.parse_args()
    return args


def _repl_loop(
    agent: Agent,
    stream_enabled: bool,
    session_manager: Any = None,
) -> None:
    """交互式 REPL 循环。"""
    # 会话启动提示
    if session_manager:
        entry = session_manager.get_or_create_session()
        if entry.was_auto_reset:
            print(f"⏳ 新会话（上次会话因 {entry.auto_reset_reason} 已过期）")
        elif entry.message_count > 0:
            title = f"「{entry.title}」" if entry.title else ""
            print(f"📋 继续上次会话 {title}（{entry.message_count} 条消息） ID: {entry.session_id}")
        else:
            print("🆕 新会话")
        print()

    while True:
        try:
            user_input = input("\n\033[1;36m>>> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            os._exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        print()
        try:
            if stream_enabled:
                _print_stream_formatted(agent.stream(user_input))
            else:
                result = agent.run(user_input)
                print(result)
        except KeyboardInterrupt:
            agent.stop()
            print("\n⏹️  Agent 已终止，你可以继续输入新指令。")
        print()


def main() -> None:
    args = parse_args()
    stream_enabled = args.stream
    repl_enabled = args.repl
    user_args = args.user_input

    Config.load()

    # Apply CLI args to Config（CLI 优先级最高，覆盖环境变量）
    if args.strict:
        Config.set("permission_policy", PermissionManager.POLICY_STRICT)
    elif args.permissive:
        Config.set("permission_policy", PermissionManager.POLICY_PERMISSIVE)
    if args.max_iterations is not None:
        Config.set("max_iterations", args.max_iterations)
    if args.session:
        Config.set("session_enabled", True)

    setup_logger("INFO", log_dir=Config.get("log_dir"))

    toolManager = ToolManager()
    toolManager.register_all()

    permission_mgr = PermissionManager(
        policy=Config.get("permission_policy"),
        tool_manager=toolManager,
    )

    # Build Harness components
    max_iterations = Config.get("max_iterations")

    hook_manager = HookManager()
    hook_manager.register_all()

    event_emitter = EventEmitter()

    EventSubscribe(event_emitter).subscribe_all()

    budget = IterationBudget(max_iterations=max_iterations)

    # 可选地构建会话管理器
    session_manager = None
    if Config.get("session_enabled"):
        db_path = Config.get("session_db_path")
        session_name = Config.get("session_name") or None
        store = SessionStore(db_path=db_path)
        policy = SessionResetPolicy(mode="none")  # 默认不自动重置，后续可配置
        session_manager = SessionManager(
            store=store,
            policy=policy,
            profile=session_name,
        )

    agent = Agent(
        llm=LlmManager.get_llm_client(),
        tool_manager=toolManager,
        permission_mgr=permission_mgr,
        hook_manager=hook_manager,
        event_emitter=event_emitter,
        budget=budget,
        session_manager=session_manager,
    )

    # 检测 REPL 模式：--repl 标志，或无参数且 stdin 是终端时默认进入
    is_repl = repl_enabled or (not user_args and sys.stdin.isatty())
    if is_repl:
        try:
            import readline  # 提供行编辑和上下键历史
        except ImportError:
            pass
        print("Kocor Agent — 输入 exit 或 Ctrl+C 退出")
        if toolManager.skill_manager:
            skills = toolManager.skill_manager.list_skills(enabled_only=True)
            slash_names = [f"/{s.name}" for s in skills
                           if s.invoke_strategy in (InvokeStrategy.SLASH, InvokeStrategy.BOTH)]
            print(f"Slash 命令: {', '.join(sorted(slash_names))}")
        print()
        _repl_loop(agent, stream_enabled, session_manager)
        return

    if user_args:
        user_input = " ".join(user_args)
    else:
        if not sys.stdin.isatty():
            user_input = sys.stdin.read().strip()
        else:
            user_input = ""

    if not user_input:
        print("用法: python -m kocor \"你的问题\"")
        print("   或: python -m kocor --stream \"你的问题\"")
        print("   或: python -m kocor --repl")
        print("   或: echo \"你的问题\" | python -m kocor")
        sys.exit(1)

    try:
        if stream_enabled and hasattr(agent, "stream"):
            _print_stream_formatted(agent.stream(user_input))
        else:
            result = agent.run(user_input)
            if result:
                print(result)
    finally:
        toolManager.mcp_manager.shutdown_all()
