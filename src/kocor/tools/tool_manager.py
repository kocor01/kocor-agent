"""工具注册与执行中心。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from kocor.config import Config
from kocor.tools.definitions import ToolDefinition
from kocor.tools.truncate import ToolOutputTruncator
from kocor.llm_provider.message import ToolCall, ToolResult
from kocor.tools.permission import PermissionManager


class ToolManager:
    """工具注册与执行中心。"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self.mcp_manager = None
        self.skill_manager = None
        self._cron_scheduler = None

    def register_builtin_tools(self) -> None:
        """向当前 ToolManager 注册内置工具（文件操作、沙盒执行、bash、cron）。"""
        from kocor.tools.toolset.read_file_tool import ReadFile
        from kocor.tools.toolset.write_file_tool import WriteFile
        from kocor.tools.toolset.patch_file_tool import PatchFile
        from kocor.tools.toolset.search_file_tool import SearchFiles
        from kocor.tools.toolset.run_python import RunPython
        from kocor.tools.toolset.bash_tool import BashTool, ProcessTool
        from kocor.tools.toolset.cron_tool import CronTool

        self.memory_store = None
        self.todo_store = None
        builtin_tools = [ReadFile, WriteFile, PatchFile, SearchFiles, RunPython, BashTool, ProcessTool]
        for tools in builtin_tools:
            self.register(tools.NAME, tools.DESCRIPTION, tools.PARAMETERS, tools.handler, tools.SAFETY_LEVEL)

        # memory 工具需要 MemoryStore，handler 延迟读取 self.memory_store
        self._register_memory_tool()
        # todo 工具需要 TodoStore，handler 延迟读取 self.todo_store
        self._register_todo_tool()
        # cron 工具
        self._register_cron_tool()

    def _register_memory_tool(self) -> None:
        """注册 memory 工具（依赖 self.memory_store，可为 None）。"""
        from kocor.tools.toolset.memory_tool import MemoryTool
        self.register(
            MemoryTool.NAME, MemoryTool.DESCRIPTION, MemoryTool.PARAMETERS,
            lambda **kw: MemoryTool.handler(store=self.memory_store, **kw),
            MemoryTool.SAFETY_LEVEL,
        )

    def _register_todo_tool(self) -> None:
        """注册 todo 工具（依赖 self.todo_store，可为 None）。"""
        from kocor.tools.toolset.todo_tool import TodoTool
        self.register(
            TodoTool.NAME, TodoTool.DESCRIPTION, TodoTool.PARAMETERS,
            lambda **kw: TodoTool.handler(store=self.todo_store, **kw),
            TodoTool.SAFETY_LEVEL,
        )

    def _register_cron_tool(self) -> None:
        """注册 cronjob 工具。"""
        from kocor.tools.toolset.cron_tool import CronTool
        from kocor.tools.toolset.cron.scheduler import CronScheduler

        if self._cron_scheduler is None:
            self._cron_scheduler = CronScheduler()

        self.register(
            CronTool.NAME, CronTool.DESCRIPTION, CronTool.PARAMETERS,
            lambda **kw: CronTool.handler(**kw),
            CronTool.SAFETY_LEVEL,
        )

    def start_cron_scheduler(self) -> None:
        """启动 cron 调度器。"""
        if self._cron_scheduler is not None:
            self._cron_scheduler.start()

    def stop_cron_scheduler(self) -> None:
        """停止 cron 调度器。"""
        if self._cron_scheduler is not None:
            self._cron_scheduler.stop()

    @property
    def cron_scheduler(self):
        """获取 cron 调度器实例。"""
        return self._cron_scheduler


    def register_all(self) -> None:
        """统一注册所有工具：内置工具 → MCP 工具 → 技能工具。"""
        self.register_builtin_tools()

        from kocor.mcp import McpManager
        self.mcp_manager = McpManager(self, Config.get("mcp_config"))
        self.mcp_manager.register_all()

        from kocor.skill import SkillManager
        self.skill_manager = SkillManager(self)
        self.skill_manager.register_all(Config.get("skills_config"), Config.get("skills_dir"))

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[..., str],
        safety_level: str = PermissionManager.SAFETY_CAUTION,
    ) -> None:
        """注册工具。

        Args:
            name: 工具名称
            description: 工具描述
            parameters: JSON Schema 参数定义
            handler: 工具处理器，接收 **kwargs 返回结果字符串
            safety_level: 安全等级
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            safety_level=safety_level,
        )
        self._handlers[name] = handler

    def get_definitions(self) -> list[ToolDefinition]:
        """返回所有工具的 ToolDefinition 列表"""
        return list(self._tools.values())

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用。

        Args:
            tool_call: 工具调用请求

        Returns:
            ToolResult: 执行结果
        """
        name = tool_call.function.name
        if name not in self._handlers:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: tool '{name}' not found",
            )

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: invalid JSON arguments for '{name}'",
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                timeout = Config.get("tool_timeout")
                future = pool.submit(self._handlers[name], **args)
                result = future.result(timeout=timeout)
            truncated = ToolOutputTruncator().truncate(str(result))
            return ToolResult(tool_call_id=tool_call.id, content=truncated)
        except PermissionError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {e}",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {type(e).__name__}: {e}",
            )
