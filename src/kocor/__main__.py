"""Kocor Agent CLI 入口。

使用:
    python -m kocor "你的问题"
    python -m kocor --stream "你的问题"
    echo "你的问题" | python -m kocor
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator

from dotenv import load_dotenv

from kocor.agent import Agent
from kocor.config import load_config
from kocor.llm_client import create_llm_client
from kocor.mcp import register_mcp_tools, shutdown_mcp_clients
from kocor.llm_provider.message import StreamChunk
from kocor.skill import SkillRegistry
from kocor.tool_registry import ToolRegistry
from kocor.tools import create_default_tools

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
        if not chunk.content:
            return
        if not self.has_content:
            print("\n\n\U0001f4ac 回答")
            print(f"{'─' * self.width}")
            self.has_content = True
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
        "user_input",
        nargs="*",
        help="用户问题",
    )
    args = parser.parse_args()
    return args.stream, args.user_input


def main() -> None:
    stream_enabled, user_args = parse_args()
    load_dotenv()
    config = load_config()
    llm = create_llm_client(config)

    toolRegistry = ToolRegistry()
    create_default_tools(toolRegistry)
    mcp_clients = register_mcp_tools(toolRegistry, config.mcp_config)

    skillRegistry = SkillRegistry(toolRegistry)
    skillRegistry.load_from_config(config.skills_config)
    skillRegistry.discover_skills(config.skills_dir)
    skillRegistry.discover_cline_skills(config.skills_dir)
    skillRegistry.register_as_tools(toolRegistry)

    agent = Agent(
        llm=llm,
        tools=toolRegistry,
        skills=skillRegistry,
        max_iterations=config.max_iterations,
    )

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
        print("   或: echo \"你的问题\" | python -m kocor")
        sys.exit(1)

    try:
        if stream_enabled and hasattr(agent, "stream"):
            _print_stream_formatted(agent.stream(user_input))
        else:
            result = agent.run(user_input)
            print(result)
    finally:
        shutdown_mcp_clients(mcp_clients)


if __name__ == "__main__":
    main()
