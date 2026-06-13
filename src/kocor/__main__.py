"""Kocor Agent CLI 入口。

使用:
    python -m kocor "你的问题"
    echo "你的问题" | python -m kocor
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from kocor.agent import Agent
from kocor.config import load_config
from kocor.llm_client import create_llm_client
from kocor.tools import create_default_tools


def main() -> None:
    """CLI 主入口"""
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
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = sys.stdin.read().strip()

    if not user_input:
        print("用法: python -m kocor \"你的问题\"")
        print("   或: echo \"你的问题\" | python -m kocor")
        sys.exit(1)

    # 运行 Agent
    result = agent.run(user_input)
    print(result)


if __name__ == "__main__":
    main()
