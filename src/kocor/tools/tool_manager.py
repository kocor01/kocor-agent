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
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self.file_state = FileStateTracker()
        self._env: LocalEnvironment | None = None
        self.mcp_manager = None
        self.skill_manager = None
        # cron worker 子进程（仅主进程持有；cron worker 子进程内的
        # ToolManager 通过 include_cron=False 跳过，避免递归 spawn）
        self._cron_worker = None
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

        Args:
            include_cron: 是否注册 cronjob 工具并创建 cron worker。
                主进程默认 True。cron worker 子进程内设 False —— 既不注册
                cronjob 工具（防递归调用），也不创建 worker（避免递归 spawn）。
            include_subagent: 是否注册 subagent 工具（子代理委派）。
                cli.py 在注册时通过 ToolManager._subagent_runner 注入运行器。
        """
        from kocor.tools.toolsets.bash_tool import BashTool, ProcessTool
        from kocor.tools.toolsets.patch_file_tool import PatchFile
        from kocor.tools.toolsets.read_file_tool import ReadFile
        from kocor.tools.toolsets.search_file_tool import SearchFiles
        from kocor.tools.toolsets.write_file_tool import WriteFile

        self.memory_store = None
        self.todo_store = None

        # 文件工具：通过闭包注入 file_state
        self.register(
            ReadFile.NAME, ReadFile.DESCRIPTION, ReadFile.PARAMETERS,
            lambda **kw: ReadFile.handler(file_state=self.file_state, **kw),
            ReadFile.SAFETY_LEVEL,
        )
        self.register(
            WriteFile.NAME, WriteFile.DESCRIPTION, WriteFile.PARAMETERS,
            lambda **kw: WriteFile.handler(file_state=self.file_state, **kw),
            WriteFile.SAFETY_LEVEL,
        )
        self.register(
            PatchFile.NAME, PatchFile.DESCRIPTION, PatchFile.PARAMETERS,
            lambda **kw: PatchFile.handler(file_state=self.file_state, **kw),
            PatchFile.SAFETY_LEVEL,
        )

        # BashTool 通过闭包注入 env，ProcessTool 保持直接注册
        self.register(
            BashTool.NAME, BashTool.DESCRIPTION, BashTool.PARAMETERS,
            lambda **kw: BashTool.handler(env=self.get_or_create_env(), **kw),
            BashTool.SAFETY_LEVEL,
        )
        self.register(
            SearchFiles.NAME, SearchFiles.DESCRIPTION, SearchFiles.PARAMETERS, SearchFiles.handler, SearchFiles.SAFETY_LEVEL,
        )
        self.register(
            ProcessTool.NAME, ProcessTool.DESCRIPTION, ProcessTool.PARAMETERS, ProcessTool.handler, ProcessTool.SAFETY_LEVEL,
        )

        # memory 工具依赖 self.memory_store，handler 在调用时读取
        from kocor.tools.toolsets.memory_tool import MemoryTool
        self.register(
            MemoryTool.NAME, MemoryTool.DESCRIPTION, MemoryTool.PARAMETERS,
            lambda **kw: MemoryTool.handler(store=self.memory_store, **kw),
            MemoryTool.SAFETY_LEVEL,
        )

        # todo 工具依赖 self.todo_store，handler 在调用时读取
        from kocor.tools.toolsets.todo_tool import TodoTool
        self.register(
            TodoTool.NAME, TodoTool.DESCRIPTION, TodoTool.PARAMETERS,
            lambda **kw: TodoTool.handler(store=self.todo_store, **kw),
            TodoTool.SAFETY_LEVEL,
        )

        # cron 工具：主进程注册 cronjob 工具 + 创建 cron worker 子进程管理器。
        # cron worker 子进程自身跳过此块（include_cron=False）。
        if include_cron:
            from kocor.tools.toolsets.cron.worker_process import CronWorkerProcess
            from kocor.tools.toolsets.cron_tool import CronTool
            if self._cron_worker is None:
                self._cron_worker = CronWorkerProcess()
            self.register(
                CronTool.NAME, CronTool.DESCRIPTION, CronTool.PARAMETERS,
                lambda **kw: CronTool.handler(**kw),
                CronTool.SAFETY_LEVEL,
            )

        # subagent 工具：注册委派子代理工具。
        # 非 orchestrator 角色（子代理自身）由 _build_child_tool_manager 管理。
        if include_subagent:
            from kocor.tools.toolsets.subagent.tool import SubagentTool
            # 在调用时解析 self._subagent_runner（cli.py 在 LLM 创建后注入），
            # 不在此处闭包绑定（避免 LLM 未就绪时 runner 不存在的问题）
            self.register(
                SubagentTool.NAME, SubagentTool.DESCRIPTION, SubagentTool.PARAMETERS,
                lambda **kw: SubagentTool.handler(runner=self._subagent_runner, **kw),
                SubagentTool.SAFETY_LEVEL,
                timeout=0,  # 豁免全局超时（子代理通常需要数分钟）
            )

    def start_cron_scheduler(self) -> None:
        """启动 cron worker 子进程。接口名保留以兼容 Agent。"""
        if self._cron_worker is not None:
            self._cron_worker.start()

    def stop_cron_scheduler(self) -> None:
        """停止 cron worker 子进程。接口名保留以兼容 Agent。"""
        if self._cron_worker is not None:
            self._cron_worker.stop()

    @property
    def cron_worker(self):
        """获取 cron worker 子进程管理器实例。"""
        return self._cron_worker


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
                defn = self._tools.get(name)
                timeout = defn.timeout if (defn is not None and defn.timeout is not None) else Config.load().tool_timeout
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
