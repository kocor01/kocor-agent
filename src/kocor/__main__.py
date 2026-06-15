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


def _print_stream_formatted(chunks):
    """格式化打印流式输出（Unicode 框线 + emoji 区块）。

    按轮次分隔，reasoning / 工具调用 / 工具结果 / 结果输出各自独立区块，
    内容增量流式打印。
    """
    round_num = 0
    tool_call_buffer: list = []
    tool_result_buffer: list = []
    has_reasoning = False
    has_content = False
    has_tool_calls = False
    has_tool_results = False

    def flush_tool_section() -> None:
        """打印工具调用和工具结果区块。"""
        nonlocal has_tool_calls, has_tool_results
        if tool_call_buffer:
            print("\n\U0001f527 工具调用")
            print("  ─" * 55)
            seen = set()
            for tc in tool_call_buffer:
                if tc.id not in seen:
                    seen.add(tc.id)
                    print(f"  • {tc.function.name}({tc.function.arguments})")
            has_tool_calls = True
        if tool_result_buffer:
            print("\n\U0001f4e6 工具结果")
            print("  ─" * 55)
            for tr in tool_result_buffer:
                content = tr.content
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"    {content}")
            has_tool_results = True

    for chunk in chunks:
        # --- 新轮次开始：打印标题 ---
        # tool_result chunk 不触发新轮次（它是上一轮工具执行的产物）
        if not chunk.tool_result and not has_reasoning and not has_content and not has_tool_calls and not has_tool_results:
            round_num += 1
            _print_round_header(round_num)

        # --- reasoning 增量 ---
        if chunk.reasoning:
            if not has_reasoning:
                print("\n\U0001f9e0 思维链")
                print("  ─" * 55)
                has_reasoning = True
            print(chunk.reasoning, end="", flush=True)

        # --- content 增量 ---
        if chunk.content:
            if not has_reasoning:
                # 没有 reasoning，先打印 reasoning 占位
                print("\n\U0001f9e0 思维链")
                print("  ─" * 55)
                has_reasoning = True
            if not has_content:
                print("\n\U0001f4dd 结果输出")
                print("  ─" * 55)
                has_content = True
            print(chunk.content, end="", flush=True)

        # --- 工具调用收集 ---
        if chunk.tool_calls:
            tool_call_buffer.extend(chunk.tool_calls)

        # --- 工具结果收集 ---
        if chunk.tool_result:
            tool_result_buffer.append(chunk.tool_result)

        # --- 本轮结束：打印工具调用 + 结果 + 重置 ---
        if chunk.is_final:
            flush_tool_section()
            # 重置状态
            has_reasoning = False
            has_content = False
            has_tool_calls = False
            has_tool_results = False
            tool_call_buffer.clear()
            tool_result_buffer.clear()


def _print_round_header(n: int) -> None:
    """打印轮次标题卡片。"""
    title = f" ⚡ 第 {n} 次请求 "
    width = 100
    bar = "─" * width
    print(f"\n╭{bar}╮")
    # 居中对齐
    padding = width - len(title) - 4
    left = padding // 2
    right = padding - left
    print(f"│{' ' * left}{title}{' ' * right}│")
    print(f"╰{bar}╯")


def parse_args():
    """解析命令行参数。

    Returns:
        (stream_enabled, user_args): 流式开关和剩余位置参数
    """
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
    """CLI 主入口"""
    # 解析参数
    stream_enabled, user_args = parse_args()

    # 加载 .env
    load_dotenv()

    # 加载配置
    config = load_config()

    # 创建 LLM 客户端
    llm = create_llm_client(config)

    # 创建工具集
    tools = create_default_tools(config)

    # 创建 Agent
    agent = Agent(llm=llm, tools=tools, max_iterations=config.max_iterations)

    # 获取用户输入
    if user_args:
        user_input = " ".join(user_args)
    else:
        # 仅在管道输入时读取 stdin，避免交互式终端永久阻塞
        if not sys.stdin.isatty():
            user_input = sys.stdin.read().strip()
        else:
            user_input = ""

    if not user_input:
        print("用法: python -m kocor \"你的问题\"")
        print("   或: python -m kocor --stream \"你的问题\"")
        print("   或: echo \"你的问题\" | python -m kocor")
        sys.exit(1)

    # 运行 Agent（流式格式化输出）
    if stream_enabled and hasattr(agent, "stream"):
        _print_stream_formatted(agent.stream(user_input))
    else:
        result = agent.run(user_input)
        print(result)


if __name__ == "__main__":
    main()
