"""工具注册与执行中心。"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from kocor.config import Config
from kocor.llm_provider.message import ToolCall, ToolResult
from kocor.tools.definitions import ToolDefinition
from kocor.tools.permission import PermissionManager
from kocor.tools.toolsets.bash.environment import LocalEnvironment
from kocor.tools.toolsets.file.file_state import FileStateTracker
from kocor.tools.truncate import ToolOutputTruncator


class ToolManager:
    """工具注册与执行中心。"""

    def __init__(self):
        # 工具定义字典：name → ToolDefinition
        self._tools: dict[str, ToolDefinition] = {}
        # 工具处理器字典：name → Callable（执行时按名查找）
        self._handlers: dict[str, Callable] = {}
        # 权限管理器：工具调用的策略决策层
        self.permission_mgr = PermissionManager(
            policy=Config.load().permission_policy,
        )
        # 文件状态追踪器：去重缓存、连续读检测、补丁失败计数
        self.file_state = FileStateTracker()
        # 本地执行环境（延迟初始化，首次 bash 调用时创建）
        self._env: LocalEnvironment | None = None
        self.mcp_manager = None
        self.skill_manager = None
        # subagent 运行器（运行时注入，由 cli.py/Agent 在 LLM 创建后设置）
        self._subagent_runner = None

    def get_or_create_env(self) -> LocalEnvironment:
        """获取或创建 LocalEnvironment 实例（延迟初始化）。

        每个 ToolManager 拥有独立的执行环境，天然支持多 Agent 隔离。
        """
        if self._env is None:
            self._env = LocalEnvironment(cwd=os.getcwd(), timeout=180)
        return self._env

    def reset_env(self) -> None:
        """重置执行环境（清理快照文件并重建）。"""
        if self._env is not None:
            self._env.cleanup()
        self._env = None

    def register_builtin_tools(
        self,
        include_cron: bool = True,
        include_subagent: bool = False,
    ) -> None:
        """向当前 ToolManager 注册内置工具（文件操作、沙盒执行、bash、cron、subagent）。

        每个工具类通过 handler_factory 类方法提供 handler 工厂，
        注入共享依赖（file_state、env、memory_store 等）。

        Args:
            include_cron: 是否注册 cronjob 工具。主进程默认 True。
                cron worker 子进程内设 False，避免递归调用。
            include_subagent: 是否注册 subagent 工具（子代理委派）。
                cli.py 在注册时通过 ToolManager._subagent_runner 注入运行器。
        """
        from kocor.tools.toolsets.bash_tool import BashTool, ProcessTool
        from kocor.tools.toolsets.cron_tool import CronTool
        from kocor.tools.toolsets.memory_tool import MemoryTool
        from kocor.tools.toolsets.patch_file_tool import PatchFileTool
        from kocor.tools.toolsets.read_file_tool import ReadFileTool
        from kocor.tools.toolsets.search_file_tool import SearchFilesTool
        from kocor.tools.toolsets.subagent.tool import SubagentTool
        from kocor.tools.toolsets.todo_tool import TodoTool
        from kocor.tools.toolsets.write_file_tool import WriteFileTool

        self.memory_store = None
        self.todo_store = None

        # 构建共享依赖字典。
        # memory_store/todo_store 在 Agent.__init__ 中设定（晚于注册），
        # 因此传 ToolManager 自身引用，由 handler 在调用时解析。
        deps = {
            "file_state": self.file_state,
            "env": self.get_or_create_env(),
            "tool_manager": self,
            "subagent_runner": self._subagent_runner,
        }

        # 始终注册的核心工具（cron/subagent 按条件跳过）
        core_tools = [
            ReadFileTool, WriteFileTool, PatchFileTool, BashTool,
            SearchFilesTool, ProcessTool, MemoryTool, TodoTool,
        ]
        if include_cron:
            core_tools.append(CronTool)
        if include_subagent:
            core_tools.append(SubagentTool)
        for tool_cls in core_tools:
            handler = tool_cls.handler_factory(**deps)
            timeout = getattr(tool_cls, 'TIMEOUT', None)
            if timeout is None:
                timeout = Config.load().tool_timeout
            self.register(
                tool_cls.NAME, tool_cls.DESCRIPTION, tool_cls.PARAMETERS,
                handler, tool_cls.SAFETY_LEVEL,
                timeout=timeout,
            )

    def register_all(self, include_subagent: bool = False) -> None:
        """统一注册所有工具：内置工具 → MCP 工具 → 技能工具。

        Args:
            include_subagent: 是否注册 subagent 工具（顶层父代理应启用）。
        """
        self.register_builtin_tools(include_subagent=include_subagent)

        from kocor.mcp import McpManager
        self.mcp_manager = McpManager(self, Config.load().mcp_config)
        self.mcp_manager.register_all()

        from kocor.skill import SkillManager
        self.skill_manager = SkillManager(self)
        self.skill_manager.register_all(Config.load().skills_config, Config.load().skills_dir)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[..., str],
        safety_level: str = PermissionManager.SAFETY_CAUTION,
        timeout: int | None = None,
    ) -> None:
        """注册工具。

        Args:
            name: 工具名称
            description: 工具描述
            parameters: JSON Schema 参数定义
            handler: 工具处理器，接收 **kwargs 返回结果字符串
            safety_level: 安全等级
            timeout: 工具级超时覆盖。None=继承 Config.tool_timeout，
                0=不超时，正数=自定义秒数（供 subagent 等长生命周期工具）
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            safety_level=safety_level,
            timeout=timeout,
        )
        self._handlers[name] = handler
        self.permission_mgr.update_safety(name, safety_level)

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
            # 使用 ThreadPoolExecutor 实现工具级超时，
            # 因为 handler 可能是同步阻塞的（如 bash 命令长时间运行）
            with ThreadPoolExecutor(max_workers=1) as pool:
                defn = self._tools.get(name)
                timeout = defn.timeout if defn is not None and defn.timeout is not None else Config.load().tool_timeout
                future = pool.submit(self._handlers[name], **args)
                result_timeout = None if timeout == 0 else timeout
                try:
                    result = future.result(timeout=result_timeout)
                except KeyboardInterrupt:
                    # 用户 Ctrl+C：通知子代理运行器停止，然后重新抛出中断
                    if self._subagent_runner is not None:
                        self._subagent_runner.stop()
                    raise
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