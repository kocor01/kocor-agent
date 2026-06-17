"""内部工具集合。

提供 create_default_tools 工厂函数，组装 read_file、write_file、run_python 等内置工具。
"""

from __future__ import annotations

from kocor.tool_registry import ToolRegistry
from kocor.tools.toolset import read_file, run_python, write_file


def create_default_tools(toolRegistry: ToolRegistry) -> None:
    """向指定 toolRegistry 注册内部工具（读文件、写文件、沙盒执行 Python）。

    Args:
        toolRegistry: 目标 ToolRegistry 实例
    """
    read_file.toolRegistry_to(toolRegistry)
    write_file.toolRegistry_to(toolRegistry)
    run_python.toolRegistry_to(toolRegistry)
