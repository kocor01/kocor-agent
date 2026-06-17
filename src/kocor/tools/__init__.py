"""内部工具集合。

提供 create_default_tools 工厂函数，组装 read_file、write_file、run_python 等内置工具。
"""

from __future__ import annotations

from kocor.config import LLMConfig
from kocor.tool_registry import ToolRegistry
from kocor.tools.toolset import read_file, run_python, write_file


def create_default_tools(config: LLMConfig | None = None) -> ToolRegistry:
    """创建默认工具集（读文件、写文件、沙盒执行 Python）。

    Args:
        config: 可选配置

    Returns:
        已注册内置工具的 ToolRegistry
    """
    timeout = config.timeout if config else 30
    registry = ToolRegistry(timeout=timeout)

    read_file.register_to(registry)
    write_file.register_to(registry)
    run_python.register_to(registry)

    return registry
