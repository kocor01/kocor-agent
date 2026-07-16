"""Kocor Agent CLI 入口。

使用:
    python -m kocor "你的问题"
    python -m kocor --no-stream "你的问题"
    python -m kocor           # 交互模式（默认流式输出、会话持久化）
    echo "你的问题" | python -m kocor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from kocor._cli.output import _print_stream_formatted, _print_welcome
from kocor.agent import Agent
from kocor.config import Config
from kocor.logger import Logger
from kocor.tools.permission import PermissionManager


def parse_args():
    parser = argparse.ArgumentParser(description="Kocor Agent - 小而美的 LLM 自主 Agent 助手")
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        default=True,
        help="禁用流式输出",
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
        "--debug",
        action="store_true",
        default=False,
        help="启用 DEBUG 日志级别",
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
) -> None:
    """交互式 REPL 循环。"""
    # 会话启动展示已由 _print_welcome 在 main 中完成

    # 注册 SIGINT handler：在 KeyboardInterrupt 之外额外调用 agent.stop()，
    # 让 _stop_requested 标志尽早设置，配合 loop 中的检查点立即停止循环。
    # 在 Windows 上，此 handler 与默认 KeyboardInterrupt 行为共存（两者都执行）；
    # 在 Unix 上，handler 替换默认行为，我们需要手动 raise KeyboardInterrupt。
    import signal

    def _sigint_handler(signum, frame):
        agent.stop()
        # Unix 上自定义 handler 替换默认 KeyboardInterrupt，需手动抛出；
        # Windows 上 Python 会额外自动抛出 KeyboardInterrupt，双重抛出无害
        # （第二次抛出在 finally 中会被抑制）。
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint_handler)

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
    user_args = args.user_input

    Config.load()

    # Apply CLI args to Config（CLI 优先级最高，覆盖环境变量）
    if args.strict:
        Config.load().permission_policy = PermissionManager.POLICY_STRICT
    elif args.permissive:
        Config.load().permission_policy = PermissionManager.POLICY_PERMISSIVE
    if args.debug:
        Config.load().log_level = "DEBUG"

    # 检测 REPL 模式：无参数且 stdin 是终端时默认进入
    is_repl = not user_args and sys.stdin.isatty()

    # 一次性模式默认关闭会话持久化
    if not is_repl:
        Config.load().session_enabled = False

    logger = Logger(Config.load().log_level, log_dir=Config.load().log_dir)

    # 使用 AgentBuilder 装配 Agent 及其所有依赖组件
    from kocor._cli.builder import AgentBuilder
    agent = (
        AgentBuilder()
        .build_llm()
        .build_subagent()
        .build_tools()
        .build_permission()
        .build_hooks(logger)
        .build_session()
        .build()
    )

    if is_repl:
        print()
        try:
            import readline  # noqa: F401  # 提供行编辑和上下键历史
        except ImportError:
            pass
        _print_welcome(agent.session_manager, agent.tool_manager.skill_manager)
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
        sys.exit(1)

    try:
        if stream_enabled and hasattr(agent, "stream"):
            _print_stream_formatted(agent.stream(user_input))
        else:
            result = agent.run(user_input)
            if result:
                print(result)
    finally:
        agent.tool_manager.mcp_manager.shutdown_all()
        agent.tool_manager.stop_cron_scheduler()

        # Debug 模式输出会话指标摘要
        if args.debug:
            cli_logger = logging.getLogger(__name__)
            metrics_data = agent.metrics
            if metrics_data:
                cli_logger.debug("Session metrics:\n%s", json.dumps(metrics_data, indent=2, ensure_ascii=False))