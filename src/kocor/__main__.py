"""Kocor Agent CLI 入口。

使用:
    python -m kocor "你的问题"
    python -m kocor --stream "你的问题"
    echo "你的问题" | python -m kocor
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from kocor.agent import Agent
from kocor.config import load_config
from kocor.llm_client import create_llm_client
from kocor.tools import create_default_tools

W = 58


def _print_stream_formatted(chunks):
    round_num = 0
    pending_round = False
    tool_calls: list = []
    has_reasoning = False
    has_content = False
    has_tool_section = False
    tool_result_idx = 0

    def _round_header(n: int):
        title = f"⚡ 第 {n} 次请求"
        fill = W - len(title) - 2
        print(f"\n── {title} {'─' * max(0, fill)}")

    for chunk in chunks:
        if not chunk.tool_result and not pending_round and not has_reasoning and not has_content and not has_tool_section:
            round_num += 1
            pending_round = True

        if pending_round and (chunk.reasoning or chunk.content or chunk.tool_calls or chunk.is_final):
            _round_header(round_num)
            pending_round = False

        # --- reasoning ---
        if chunk.reasoning:
            if not has_reasoning:
                print(f"\n\U0001f9e0 思维过程")
                print(f"{'─' * W}")
                has_reasoning = True
            print(chunk.reasoning, end="", flush=True)

        # --- content ---
        if chunk.content:
            if not has_reasoning:
                print(f"\n\n\U0001f9e0 思维过程")
                print(f"{'─' * W}")
                has_reasoning = True
            if not has_content:
                print(f"\n\n\U0001f4ac 回答")
                print(f"{'─' * W}")
                has_content = True
            print(chunk.content, end="", flush=True)

        # --- tool calls ---
        if chunk.tool_calls:
            seen_ids = {tc.id for tc in tool_calls}
            for tc in chunk.tool_calls:
                if tc.id not in seen_ids:
                    tool_calls.append(tc)
                    seen_ids.add(tc.id)
            if not has_tool_section and not chunk.tool_result:
                print(f"\n\n\U0001f527 工具调用")
                print(f"{'─' * W}")
                has_tool_section = True

        # --- tool results ---
        if chunk.tool_result:
            if tool_result_idx < len(tool_calls):
                tc = tool_calls[tool_result_idx]
                content = chunk.tool_result.content
                if len(content) > 200:
                    content = content[:200] + "..."
                if tool_result_idx > 0:
                    print()
                print(f"{tool_result_idx + 1:2}. {tc.function.name}({tc.function.arguments})")
                print(f"{'─' * (W - 4)}")
                print(f"{content}")
                tool_result_idx += 1

        # --- end of round ---
        if chunk.is_final:
            has_reasoning = False
            has_content = False
            has_tool_section = False
            if chunk.tool_result and tool_result_idx >= len(tool_calls) and tool_calls:
                tool_calls.clear()
                tool_result_idx = 0

    for tc in tool_calls[tool_result_idx:]:
        print(f"• {tc.function.name}({tc.function.arguments})")


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
    tools = create_default_tools(config)
    agent = Agent(llm=llm, tools=tools, max_iterations=config.max_iterations)

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

    if stream_enabled and hasattr(agent, "stream"):
        _print_stream_formatted(agent.stream(user_input))
    else:
        result = agent.run(user_input)
        print(result)


if __name__ == "__main__":
    main()
