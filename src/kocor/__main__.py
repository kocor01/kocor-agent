"""Kocor Agent CLI 入口。

使用:
    python -m kocor "你的问题"
    python -m kocor --stream "你的问题"
    python -m kocor --repl           # 交互模式
    echo "你的问题" | python -m kocor
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator

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
        self.content_emitted = False

    def _round_header(self, n: int) -> None:
        title = f"⚡ 第 {n} 次请求"
        fill = self.width - 2 - len(title)
        print(f"\n── {title} {'─' * max(0, fill)}")

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
        if not chunk.content or not chunk.content.strip():
            return
        if not self.has_content:
            print("\n\U0001f4ac 回答")
            print(f"{'─' * self.width}")
            self.has_content = True
            self.content_emitted = False
        if not self.content_emitted:
            self.content_emitted = True
            print(chunk.content.lstrip("\n"), end="", flush=True)
        else:
            print(chunk.content, end="", flush=True)

    def _handle_tool_calls(self, chunk) -> None:
        if not chunk.tool_calls:
            return
        seen_ids = {tc.id for tc in self.tool_calls}
        for tc in chunk.tool_calls:
            if tc.id not in seen_ids:
                self.tool_calls.append(tc)
                seen_ids.add(tc.id)
        if not self.has_tool_section and not chunk.tool_result:
            print("\n\n\U0001f527 工具调用")
            print(f"{'─' * self.width}")
            self.has_tool_section = True

    def _handle_tool_results(self, chunk) -> None:
        if not chunk.tool_result:
            return
        if self.tool_result_idx < len(self.tool_calls):
            tc = self.tool_calls[self.tool_result_idx]
            content = chunk.tool_result.content
            if len(content) > 200:
                content = content[:200] + "..."
            if self.tool_result_idx > 0:
                print()
            print(f"{self.tool_result_idx + 1:2}. {tc.function.name}({tc.function.arguments})")
            print(f"{'─' * (self.width - 4)}")
            print(f"{content}")
            self.tool_result_idx += 1

    def _handle_end_of_round(self, chunk) -> None:
        if not chunk.is_final:
            return
        self.has_reasoning = False
        self.has_content = False
        self.has_tool_section = False
        self.content_emitted = False
        if chunk.tool_result and self.tool_result_idx >= len(self.tool_calls) and self.tool_calls:
            self.tool_calls.clear()
            self.tool_result_idx = 0

    def flush_remaining(self) -> None:
        for tc in self.tool_calls[self.tool_result_idx:]:
            print(f"• {tc.function.name}({tc.function.arguments})")


def _print_stream_formatted(chunks: Iterator[StreamChunk]) -> None:
    formatter = _StreamFormatter()
    for chunk in chunks:
        formatter.handle_chunk(chunk)
    formatter.flush_remaining()


def parse_args():
    parser = argparse.ArgumentParser(description="Kocor Agent - LLM 自主 Agent 助手")
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
        "user_input",
        nargs="*",
        help="用户问题",
    )
    args = parser.parse_args()
    return args


def _repl_loop(agent: Agent, stream_enabled: bool) -> None:
    """交互式 REPL 循环。"""
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        print()
        if stream_enabled:
            _print_stream_formatted(agent.stream(user_input))
        else:
            result = agent.run(user_input)
            print(result)
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

    setup_logger("INFO")

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

    budget = IterationBudget(iterations_limit=max_iterations)

    agent = Agent(
        llm=LlmManager.get_llm_client(),
        tool_manager=toolManager,
        skill_manager=toolManager.skill_manager,
        max_iterations=max_iterations,
        permission_mgr=permission_mgr,
        hook_manager=hook_manager,
        event_emitter=event_emitter,
        budget=budget,
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
        _repl_loop(agent, stream_enabled)
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
            print(result)
    finally:
        toolManager.mcp_manager.shutdown_all()


if __name__ == "__main__":
    main()
