# Kocor Agent Harness（驾驭工程）技术方案

> **文档版本**: v0.1  
> **最后更新**: 2026-06-21  
> **关联设计**: [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) — 整体技术方案 | [context_design.md](context_design.md) — 上下文管理设计

---

## 目录

1. [背景与目标](#一背景与目标)
2. [行业调研](#二行业调研)
3. [核心概念](#三核心概念)
4. [架构总览](#四架构总览)
5. [Agent 核心循环](#五agent-核心循环)
6. [工具系统](#六工具系统)
7. [权限系统](#七权限系统)
8. [上下文管理](#八上下文管理)
9. [扩展系统](#九扩展系统)
10. [沙盒与安全](#十沙盒与安全)
11. [可观测性](#十一可观测性)
12. [错误处理与恢复](#十二错误处理与恢复)
13. [配置系统](#十三配置系统)
14. [CLI 与交互](#十四cli-与交互)
15. [实施路径](#十五实施路径)

---

## 一、背景与目标

### 1.1 什么是 Harness（驾驭工程）

Harness 是 Agent 的"驾驶舱"—— 它不是 Agent 本身，而是**承载、驱动、约束和观察 Agent 的运行时系统**。一个完整的 Harness 包含：

```
┌─────────────────────────────────────────────┐
│              Harness（驾驭工程）               │
│                                             │
│  入口层  ── CLI / API / IDE 集成             │
│                                             │
│  调度层  ── Agent 循环、迭代控制、生命周期      │
│                                             │
│  工具层  ── 注册、执行、权限、沙盒             │
│                                             │
│  上下文层 ── System Prompt、记忆、历史管理     │
│                                             │
│  扩展层  ── Skills、MCP、Hooks               │
│                                             │
│  观测层  ── 日志、调试、Token 审计            │
└─────────────────────────────────────────────┘
```

### 1.2 当前现状

Kocor Agent 已有不错的骨架，但在 Harness 维度还有以下缺口：

| 领域 | 当前状态 | 缺口 |
|------|---------|------|
| Agent 循环 | 基本的 `generate → execute → loop` | 无中断恢复、无迭代预算管理 |
| 工具系统 | ToolRegistry 注册机制 | 权限粒度粗、无分组、无流控 |
| 权限系统 | MCP 权限检查（always_allow/ask） | 未覆盖内置工具、无持久化 |
| 上下文 | 分层 System Prompt、记忆系统 | 摘要/滑动窗口/Token 预算未集成到循环 |
| 扩展 | Skills + MCP | 无 Hooks 机制、无生命周期事件 |
| 沙盒 | subprocess + timeout | 无资源限制、无网络隔离 |
| 可观测性 | 无 | 无日志、无调试模式、无 Token 审计 |
| 配置 | 环境变量 + JSON | 配置分散、缺乏验证 |

### 1.3 设计目标

| 维度 | 目标 |
|------|------|
| **安全性** | 所有工具调用可控、可审计、可撤销 |
| **可观测性** | 每一次迭代、每一个 token、每一个工具调用都可追溯 |
| **可扩展性** | 通过 Skills + MCP + Hooks 三机制可扩展 |
| **鲁棒性** | 优雅降级、错误恢复、预算保护 |
| **透明性** | Agent 知道自己的上下文边界和预算状态 |
| **轻量性** | 保持"小而美"定位，不引入重型框架 |

### 1.4 设计原则

1. **安全默认**：所有敏感操作默认需要确认，除非用户明确授权
2. **显式优于隐式**：Agent 的决策过程对用户可见
3. **分层关注**：每一层只做一件事，层间通过清晰接口通信
4. **渐进复杂**：简单场景零配置工作，复杂场景可精细控制
5. **最小依赖**：核心循环零第三方依赖，扩展点通过接口而非继承

---

## 二、行业调研

### 2.1 Claude Code（Anthropic）— 基准级 Harness 设计

Claude Code 是目前最成熟的 Agent Harness 实现之一，可直接观察到其架构特征。

**核心设计：**

```
┌──────────────────────────────────────────────┐
│                   CLI 入口                     │
│  claude "question" | claude /command          │
├──────────────────────────────────────────────┤
│              Agent Loop Harness                │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  System Prompt 构建                       │  │
│  │  ├── Core Identity（固定）                │  │
│  │  ├── CLAUDE.md（项目指令）                │  │
│  │  ├── Memory Block（持久记忆）             │  │
│  │  ├── Environment（git、CWD、OS）          │  │
│  │  └── Session History（增长中）            │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  工具执行 Harness                         │  │
│  │  ├── Permission Check ──→ allow/ask/deny  │  │
│  │  ├── Tool Execution ──→ subprocess/API    │  │
│  │  ├── Output Truncation ──→ 三级截断       │  │
│  │  └── Result Injection ──→ back to loop    │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  Hooks 系统                               │  │
│  │  ├── Pre-tool hook                        │  │
│  │  ├── Post-tool hook                       │  │
│  │  ├── Pre-message hook                     │  │
│  │  └── Post-message hook                    │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  MCP 扩展系统                             │  │
│  │  └── 外部工具服务器桥接                    │  │
│  └──────────────────────────────────────────┘  │
├──────────────────────────────────────────────┤
│              设置 / 配置系统                    │
│  settings.json / .claude/settings.local.json  │
├──────────────────────────────────────────────┤
│              记忆 / 持久化                     │
│  ~/.claude/memories/*.md + MEMORY.md 索引      │
└──────────────────────────────────────────────┘
```

**关键设计特征：**

| 特征 | 实现方式 | 可借鉴程度 |
|------|---------|-----------|
| 多层 System Prompt | 6 层分层合并 | ⭐⭐⭐⭐⭐ 直接采纳 |
| 文件记忆系统 | Markdown + YAML frontmatter + 索引 | ⭐⭐⭐⭐⭐ 已部分实现 |
| Hooks 系统 | settings.json 中声明式配置 | ⭐⭐⭐⭐ 核心扩展机制 |
| 工具权限 | 三级权限（always/ask/deny）+ 会话缓存 | ⭐⭐⭐⭐⭐ 需扩展 |
| 上下文感知 | git 状态、当前文件、最近编辑自动注入 | ⭐⭐⭐⭐ 已有 env_info |
| 配置分层 | 全局默认 → 项目设置 → 本地覆盖 | ⭐⭐⭐⭐ 可借鉴 |
| 沙盒执行 | 子进程隔离 + 超时 + 环境变量过滤 | ⭐⭐⭐ 当前已有雏形 |

### 2.2 Hermes Agent（Nous Research）— 函数调用 Router

**核心设计：**

```
┌──────────────────────────────────────┐
│           Hermes Router Agent         │
│                                       │
│  Query ──→ Intent Classifier ──→      │
│              │                        │
│              ├── Tool Selector        │
│              │   └─ 从工具库中选出     │
│              │      最相关的 N 个工具  │
│              │                        │
│              └── Function Caller      │
│                  └─ 执行选定工具       │
│                                       │
│  Key: 动态工具选择减少 token 消耗      │
└──────────────────────────────────────┘
```

**关键启发：**

| 特征 | 说明 | 可借鉴程度 |
|------|------|-----------|
| 动态工具选择 | 根据意图只注入相关工具 | ⭐⭐⭐⭐ 后续阶段 |
| 意图路由 | 先分类再执行 | ⭐⭐⭐ 可选增强 |
| 结构化输出 | 强制 JSON Schema 输出 | ⭐⭐⭐⭐⭐ 可直接采纳 |

### 2.3 Cline（VS Code 扩展）

**核心设计：**

```
┌──────────────────────────────────────────┐
│              Cline Agent                  │
│                                           │
│  上下文管理:                               │
│  ├── Token Budget 跟踪（实时）             │
│  ├── 70% 阈值 → 触发摘要                  │
│  ├── 90% 阈值 → 强制截断                  │
│  └── 滑动窗口保留最近 N 轮                 │
│                                           │
│  权限管理:                                 │
│  ├── 每次工具调用前弹出确认对话框          │
│  ├── 同一工具可"始终允许此会话"           │
│  └── 敏感操作默认需要确认                  │
│                                           │
│  模式系统:                                 │
│  ├── Plan 模式（先规划再执行）            │
│  └── Act 模式（直接执行）                  │
└──────────────────────────────────────────┘
```

**关键启发：**

| 特征 | 说明 | 可借鉴程度 |
|------|------|-----------|
| Token 预算管理 | 实时跟踪 + 阈值触发 | ⭐⭐⭐⭐⭐ 已在 context 计划内 |
| Plan/Act 模式 | 先规划再执行 | ⭐⭐⭐ 可选增强 |
| 用户确认流程 | 每次操作前确认 | ⭐⭐⭐⭐ 扩展已有权限 |
| 工具输出截断 | 过长输出自动截断 | ⭐⭐⭐⭐⭐ 已在 MCP 实现 |

### 2.4 Aider — 按需加载与 Repo Map

**核心设计：**

```
┌──────────────────────────────────────┐
│            Aider Agent                │
│                                       │
│  Repo Map:                            │
│  ├── 代码库结构压缩地图                │
│  ├── 关键符号索引                      │
│  └── 自动随对话更新                    │
│                                       │
│  按需加载:                             │
│  ├── 只加载任务相关文件到上下文        │
│  └── 工具输出懒加载                    │
│                                       │
│  Architect/Editor 双模式:             │
│  ├── Architect: 设计架构              │
│  └── Editor: 修改代码                  │
└──────────────────────────────────────┘
```

**关键启发：**

| 特征 | 说明 | 可借鉴程度 |
|------|------|-----------|
| Repo Map | 代码结构概览 | ⭐⭐⭐ 可选增强 |
| 按需加载 | 任务相关的文件选择 | ⭐⭐⭐ 后续阶段 |

### 2.5 综合总结

所有主流 Agent Harness 的共同模式：

```
核心循环:     Think → Act → Observe → Loop
安全控制:     工具调用 → 权限检查 → 执行 → 截断 → 注入
上下文管理:   分层提示 → 注入记忆 → 历史摘要 → 窗口控制
扩展机制:     Hooks / MCP / Plugin 三种方式
用户体验:     流式输出 / 进度展示 / 透明决策
```

Kocor Agent 应吸收这些共同模式，同时保持"小而美"的定位。

---

## 三、核心概念

### 3.1 关键术语

| 术语 | 定义 |
|------|------|
| **Harness** | 承载 Agent 的运行时系统：调度、安全、上下文、扩展 |
| **Agent Loop** | `generate → execute → observe → repeat` 核心循环 |
| **Iteration** | 循环中的单次 `generate → execute` 周期 |
| **Session** | 从用户输入到最终回复的完整调用（可能包含多次迭代）|
| **Turn** | 多轮对话中的一次用户输入 + Agent 响应 |
| **Tool** | Agent 可调用的外部能力（文件、代码、API） |
| **Skill** | 预设的提示/代码模板，可通过 `/` 或 LLM 触发 |
| **Hook** | 生命周期事件的自定义回调（pre/post tool 等）|
| **MCP** | Model Context Protocol，外部工具服务器协议 |
| **Budget** | Token、迭代次数、执行时间的上限和消耗 |

### 3.2 Harness 核心模型

```python
@dataclass
class HarnessConfig:
    """Harness 全局配置。"""
    # Agent 循环
    max_iterations: int = 20
    max_tokens_per_response: int = 4096
    
    # 权限
    permission_policy: str = "default"  # default | strict | permissive
    
    # 上下文
    context_strategy: str = "default"   # default | sliding | aggressive
    context_max_tokens: int = 200_000
    preserve_rounds: int = 3
    
    # 安全
    sandbox_timeout: int = 30
    sandbox_memory_limit: str = "256m"


@dataclass 
class IterationBudget:
    """迭代预算：持续追踪关键资源的消耗。"""
    iterations_used: int = 0
    iterations_limit: int = 20
    
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_limit: int = 200_000
    
    time_start: float = 0.0
    time_elapsed: float = 0.0
    time_limit: float = 300.0  # 5 min
    
    @property
    def exhausted(self) -> bool:
        return (self.iterations_used >= self.iterations_limit 
                or self.tokens_prompt >= self.tokens_limit
                or self.time_elapsed >= self.time_limit)
    
    @property
    def remaining_iterations(self) -> int:
        return self.iterations_limit - self.iterations_used


@dataclass
class ToolCallRecord:
    """工具调用记录：不可变的审计跟踪条目。"""
    iteration: int
    tool_name: str
    arguments: dict
    result_summary: str  # 截断后的结果摘要
    result_token_count: int
    duration_ms: float
    permission: str  # "auto" | "confirm" | "denied"
    error: str | None = None


@dataclass
class HarnessEvent:
    """Harness 事件：通过观察者/钩子分发的运行时事件。"""
    type: str  # "pre_tool" | "post_tool" | "pre_generate" | "post_generate" | "error"
    iteration: int
    data: dict
    timestamp: float = 0.0
```

---

## 四、架构总览

### 4.1 完整架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI 入口 (__main__.py)                       │
│            python -m kocor "..." | --stream                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                     Harness（调度层）                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Agent Loop Controller                        │  │
│  │                                                               │  │
│  │  1. context_builder.build_context(user_input)                 │  │
│   │  2. 检查 Budget（迭代数/Token/耗时）                         │  │
│  │  3. llm.generate() / stream()                                │  │
│  │  4. 提取 tool_calls                                          │  │
│  │  5. 对每个 tool_call:                                         │  │
│  │     a. 权限检查 (PermissionManager)                           │  │
│  │     b. 执行 (ToolRegistry.execute → handler)                 │  │
│  │     c. 输出截断 (ToolOutputTruncator)                        │  │
│  │     d. 注入回 messages                                        │  │
│  │  6. 重复直到完成或 Budget 耗尽                                │  │
│  │                                                               │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │  │
│  │  │ Event Emitter│ │ Hook Runner  │ │ Budget Keeper│          │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               Context Builder（上下文层）                      │  │
│  │  L1 身份提示 ─ L2 项目指令 ─ L3 环境 ─ L4 记忆 ─ L5 历史    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │             Tool Executor（工具执行层）                        │  │
│  │  Permission ─→ Execution ─→ Truncation ─→ Injection          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │             Extension Manager（扩展层）                       │  │
│  │  Skills Registry │ MCP Client Pool │ Hooks System            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                         LLM Provider 层                             │
│           OpenAIClient │ AnthropicClient │ (Future providers)       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 模块依赖关系

```
__main__.py (CLI)
    │
    ├──→ config.py (配置加载)
    ├──→ agent.py (Agent 类) ──→ ToolRegistry
    │                              ├── MCP tools (registration.py)
    │                              └── Built-in tools (tools/toolset/)
    │       │
    │       ├──→ context/builder.py ──→ context/memory.py
    │       │                              ├── token_counter.py
    │       │                              ├── summarizer.py
    │       │                              ├── sliding_window.py
    │       │                              └── strategies.py
    │       │
    │       ├──→ llm_client.py ──→ llm_provider/*.py
    │       │
    │       └──→ skill/registry.py ──→ skill/models.py
    │
    └──→ mcp/ (注册、权限、截断)
```

### 4.3 核心数据流

```
用户输入 "帮我分析这个项目"
        │
        ▼
1. Agent.run("帮我分析这个项目")
        │
        ▼
2. ContextBuilder.build_context(user_input, session_history)
   ├── 加载 L1 身份提示
   ├── 加载 L2 项目指令 (KOCOR.md)
   ├── 收集 L3 环境信息 (git, OS, CWD)
   ├── 加载 L4 持久记忆
   ├── 处理 L5 会话历史 (摘要/滑动窗口)
   └── 组装 messages = [system, ...history..., user]
        │
        ▼
3. Harness Loop: for _ in range(max_iterations):
        │
        ▼
4. BudgetChecker: iterations++, tokens++, time++
   └── 如果 Budget 耗尽 → 强制结束
        │
        ▼
5. EventEmitter.fire("pre_generate", iteration, messages)
   └── HookRunner.run_hooks("pre_generate")
        │
        ▼
6. llm.generate(messages, tools=tool_definitions)
        │
        ▼
7. EventEmitter.fire("post_generate", iteration, response)
        │
        ▼
8. 检查 response:
   ├── 无 tool_calls → 返回 response.content ✅
   │
   └── 有 tool_calls → 迭代处理:
        │
        ▼
9. 对每个 tool_call:
   a. 权限检查 (PermissionManager)
      ├── 允许 → 继续
      ├── 拒绝 → 返回错误消息
      └── 询问 → 等待用户确认
        │
        ▼
   b. EventEmitter.fire("pre_tool", iteration, tool_call)
        │
        ▼
   c. ToolRegistry.execute(tool_call)
      ├── 内置工具 → 直接执行
      ├── Skill 工具 → skill_registry.execute()
      └── MCP 工具 → MCPClient.call_tool()
        │
        ▼
   d. ToolOutputTruncator.truncate(result.content)
        │
        ▼
   e. EventEmitter.fire("post_tool", iteration, result)
        │
        ▼
   f. 注入 messages: Message(role="tool", ...)
        │
        ▼
10. 回到步骤 3 (下一轮迭代)
        │
        ▼
11. Budget 耗尽 → 返回超时消息（带已有结果）
```

---

## 五、Agent 核心循环

### 5.1 循环控制器设计

当前 `Agent.run()` 和 `Agent.stream()` 的逻辑嵌入在 Agent 类中，需要提取为独立的循环控制器。

```python
class AgentLoop:
    """Agent 循环控制器：管理完整的 think-act-observe 生命周期。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        context_builder: ContextBuilder,
        permission_mgr: PermissionManager,
        hook_runner: HookRunner | None = None,
        event_emitter: EventEmitter | None = None,
        budget: IterationBudget | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.context_builder = context_builder
        self.permission_mgr = permission_mgr
        self.hook_runner = hook_runner
        self.event_emitter = event_emitter or EventEmitter()
        self.budget = budget or IterationBudget()
        self._iteration = 0
        self._tool_history: list[ToolCallRecord] = []

    def run(self, user_input: str) -> str:
        """同步运行完整的 Agent 循环。"""
        # 1. 构建上下文
        context = self.context_builder.build_context(
            user_input=user_input,
            session_history=[],
        )
        messages = context.session_messages

        # 2. 核心循环
        while not self.budget.exhausted:
            self._iteration += 1
            self.budget.iterations_used = self._iteration

            # Pre-generate hook
            self._emit("pre_generate", iteration=self._iteration, messages=messages)

            # LLM 调用
            response = self.llm.generate(
                messages,
                tools=self.tools.get_definitions(),
            )
            messages.append(response)

            # Post-generate hook
            self._emit("post_generate", iteration=self._iteration, response=response)

            # 检查是否完成
            if not response.tool_calls:
                return response.content

            # 执行工具调用
            for tool_call in response.tool_calls:
                result = self._execute_one_tool(tool_call, messages)
                if result is None:  # 权限拒绝
                    continue
                messages.append(result)

        # Budget 耗尽
        return self._budget_exhausted_message(messages)

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式运行完整的 Agent 循环。"""
        context = self.context_builder.build_context(
            user_input=user_input,
            session_history=[],
        )
        messages = context.session_messages

        while not self.budget.exhausted:
            self._iteration += 1
            self.budget.iterations_used = self._iteration

            # ... 流式调用 + 分段 yield ...

            for chunk in self.llm.stream(messages, ...):
                yield chunk
                if chunk.is_final:
                    # 处理 tool_calls
                    if accumulated_tool_calls:
                        for tc in accumulated_tool_calls:
                            result = self._execute_one_tool(tc, messages)
                            ...
                    else:
                        return  # 最终答案

        yield StreamChunk(content=self._budget_exhausted_message(messages))

    def _execute_one_tool(
        self, tool_call: ToolCall, messages: list[Message]
    ) -> Message | None:
        """执行一个工具调用（含权限检查 + 截断 + 审计）。"""
        # 1. 权限检查
        if not self.permission_mgr.check(tool_call.function.name):
            # 权限拒绝
            record = ToolCallRecord(
                iteration=self._iteration,
                tool_name=tool_call.function.name,
                arguments={},
                result_summary="[Permission Denied]",
                result_token_count=0,
                duration_ms=0,
                permission="denied",
            )
            self._tool_history.append(record)
            return Message(
                role="tool",
                content="[Permission Denied] 用户拒绝了此工具调用",
                tool_call_id=tool_call.id,
            )

        # 2. Pre-tool hook
        self._emit("pre_tool", iteration=self._iteration, tool_call=tool_call)

        # 3. 执行
        start = time.monotonic()
        try:
            result = self.tools.execute(tool_call)
            duration = (time.monotonic() - start) * 1000
            truncated = self._truncate_tool_output(result.content)

            # 记录
            self._tool_history.append(ToolCallRecord(
                iteration=self._iteration,
                tool_name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments or "{}"),
                result_summary=truncated[:200],
                result_token_count=estimate_tokens(truncated),
                duration_ms=duration,
                permission="auto",
            ))

            # Post-tool hook
            self._emit("post_tool", iteration=self._iteration, result=result)

            return Message(
                role="tool",
                content=truncated,
                tool_call_id=result.tool_call_id,
            )

        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            self._tool_history.append(ToolCallRecord(
                iteration=self._iteration,
                tool_name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments or "{}"),
                result_summary=f"Error: {type(e).__name__}",
                result_token_count=0,
                duration_ms=duration,
                permission="auto",
                error=str(e),
            ))
            return Message(
                role="tool",
                content=f"Error: {type(e).__name__}: {e}",
                tool_call_id=tool_call.id,
            )

    def get_tool_history(self) -> list[ToolCallRecord]:
        """获取本次会话的工具调用历史（审计用）。"""
        return list(self._tool_history)

    def _budget_exhausted_message(self, messages: list[Message]) -> str:
        """Budgt 耗尽时的消息（含已有结果摘要）。"""
        parts = [
            f"Agent 在 {self._iteration} 次迭代后未完成。",
            f"已执行 {len(self._tool_history)} 个工具调用。",
        ]
        if self._tool_history:
            parts.append("已完成的操作:")
            for rec in self._tool_history:
                parts.append(f"  {rec.iteration}. {rec.tool_name}()")
        return "\n".join(parts)

    def _truncate_tool_output(self, content: str) -> str:
        """截断工具输出。"""
        if len(content) > 50_000:
            return content[:25_000] + "\n\n...[truncated]...\n\n" + content[-25_000:]
        if len(content.splitlines()) > 2_000:
            lines = content.splitlines()
            return "\n".join(lines[:1_000] + ["...[truncated lines]..."] + lines[-1_000:])
        return content

    def _emit(self, event_type: str, **data) -> None:
        """发射 Harness 事件。"""
        self.event_emitter.fire(HarnessEvent(
            type=event_type,
            iteration=self._iteration,
            data=data,
            timestamp=time.time(),
        ))


# -- 集成 Agent 类 --

class Agent:
    """Agent 类（精简后：作为 Harness 的面向用户的 API 外观）。"""

    def __init__(self, ..., agent_loop: AgentLoop | None = None):
        self.loop = agent_loop or self._build_default_loop(...)

    def run(self, user_input: str) -> str:
        if self.skills and user_input.startswith("/"):
            return self._handle_slash_command(user_input)
        return self.loop.run(user_input)

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        if self.skills and user_input.startswith("/"):
            result = self._handle_slash_command(user_input)
            yield StreamChunk(content=result, is_final=True)
            return
        yield from self.loop.stream(user_input)
```

### 5.2 关键设计决策

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 循环控制器的位置 | Agent 类 vs 独立类 | **独立 `AgentLoop` 类** | 关注点分离，可测试性 |
| ToolCallRecord 是否持久化 | 是/否 | **不持久化，仅在内存** | 保持轻量，审计由上层负责 |
| Budget 耗尽时返回什么 | 抛异常 vs 返回消息 | **返回带已有结果的消息** | 优雅降级，不丢失已做工作 |
| 事件发射是同步还是异步 | 同步 vs 异步 | **同步** | 保持简单，Hooks 也是同步 |

---

## 六、工具系统

### 6.1 工具类型

```
工具类型体系:

Tool (抽象)
├── BuiltinTool ── read_file, write_file, run_python
├── MCPTool ──── 来自 MCP 服务器的工具（运行时动态注册）
└── SkillTool ── 作为工具暴露的 Skill（skill_<name>）
```

### 6.2 当前 ToolRegistry 增强

```python
class ToolRegistry:
    """工具注册与执行中心（增强版）。"""

    def __init__(self, allowed_dir: str = "", timeout: int = 30):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._tool_metadata: dict[str, ToolMetadata] = {}
        self._timeout = timeout

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[..., str],
        *,
        category: str = "general",        # general | mcp | skill | filesystem
        dangerous: bool = False,          # 是否标记为危险操作
        rate_limit: int = 0,              # 每分钟最大调用次数（0=不限）
    ) -> None:
        """注册工具（增强版）。"""
        ...

    def get_definitions(
        self,
        filter_category: str | None = None,
    ) -> list[ToolDefinition]:
        """获取工具定义（支持按分类过滤）。"""
        ...

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具（含增强的错误信息）。"""
        ...

    # -- 新增方法 --

    def get_metadata(self, name: str) -> ToolMetadata | None:
        """获取工具元数据。"""
        ...

    def group_by_category(self) -> dict[str, list[str]]:
        """按分类分组。"""
        ...

    def count_tools(self) -> int:
        """返回工具总数。"""
        return len(self._tools)


@dataclass
class ToolMetadata:
    """工具元数据（增强信息）。"""
    name: str
    category: str             # general | mcp | skill | filesystem
    dangerous: bool           # 是否为危险操作
    rate_limit: int           # 速率限制
    calls_this_minute: int = 0  # 当前分钟调用次数
```

### 6.3 工具执行管道

```
工具调用 → 权限检查 → 速率限制 → 执行 → 输出截断 → 结果注入
    │           │           │         │        │           │
    │     PermissionMgr  RateLimiter  Handler  Truncator   Message
    ▼
  ┌────────────────────────────────────────────────────────────┐
  │  Pipeline（可组合的执行链）                                  │
  │                                                            │
  │  class ToolExecutionPipeline:                              │
  │      def execute(self, tool_call) -> ToolResult:           │
  │          step1: 权限检查 (PermissionManager)               │
  │          step2: 速率限制 (RateLimiter)                      │
  │          step3: 执行 (handler)                             │
  │          step4: 输出截断 (ToolOutputTruncator)             │
  │          step5: 审计记录 (ToolCallRecord)                  │
  └────────────────────────────────────────────────────────────┘
```

### 6.4 工具安全管理

```python
@dataclass
class ToolSafetyLevel:
    """工具安全等级。"""
    level: str  # safe | caution | dangerous

# 安全等级定义：
TOOL_SAFETY_MAP = {
    "read_file":      ToolSafetyLevel("caution"),    # 读取可能包含敏感信息
    "write_file":     ToolSafetyLevel("dangerous"),  # 修改文件系统
    "run_python":     ToolSafetyLevel("dangerous"),  # 执行代码
    # MCP 工具：由服务器声明
    # Skill 工具：由技能声明
}
```

### 6.5 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 工具分组 | **分类标签（category）** | 比层级结构更灵活 |
| 工具注册方式 | **保持当前装饰器/直接注册** | 简单直接 |
| 速率限制 | **内置支持，默认不限** | 防止失控调用 |
| 执行管道 | **Pipeline 模式（组合）** | 灵活扩展每个环节 |

---

## 七、权限系统

### 7.1 权限模型

```
权限策略:
┌─────────────────────────────────────────────────────────────┐
│ PermissionManager                                           │
│                                                             │
│  策略:                                                      │
│  ├── permissive (宽松)                                      │
│  │   ├── 安全工具 (read-only) → 自动放行                    │
│  │   └── 危险工具 (write/exec) → 首次询问，会话内缓存       │
│  │                                                          │
│  ├── default (默认)                                          │
│  │   ├── 安全工具 → 自动放行                                │
│  │   ├── 危险工具 → 每次询问                               │
│  │   └── MCP 工具 → 按服务器策略                           │
│  │                                                          │
│  └── strict (严格)                                          │
│       ├── 所有工具 → 每次询问                               │
│       └── 高危操作 (shell/exec) → 默认拒绝，需要显式 allow │
│                                                             │
│  策略来源:                                                  │
│  ├── 全局配置 (settings)                                    │
│  ├── 项目配置 (KOCOR.md)                                   │
│  └── 运行时参数 (--dangerous)                               │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 当前权限系统扩展

当前 `PermissionManager` 仅用于 MCP 工具，需扩展覆盖所有工具：

```python
class PermissionManager:
    """统一权限管理器（覆盖所有工具类型）。"""

    def __init__(
        self,
        policy: str = "default",       # permissive | default | strict
        always_allow: set[str] | None = None,   # 始终允许的工具名
        always_ask: set[str] | None = None,     # 始终询问的工具名
        cache_enabled: bool = True,
    ):
        self.policy = policy
        self._always_allow = always_allow or set()
        self._always_ask = always_ask or set()
        self._cache: set[str] = set()
        self.cache_enabled = cache_enabled

    def check(self, tool_name: str) -> bool:
        """检查工具调用是否需要用户确认。

        Returns:
            True = 允许执行
            False = 拒绝执行
        """
        # 1. 始终允许列表
        if tool_name in self._always_allow:
            return True

        # 2. 会话缓存
        if self.cache_enabled and tool_name in self._cache:
            return True

        # 3. 按安全检查
        safety = TOOL_SAFETY_MAP.get(tool_name, ToolSafetyLevel("caution"))

        if self.policy == "permissive":
            if safety.level == "dangerous":
                return self._ask_user(tool_name)
            return True  # safe/caution 自动放行

        if self.policy == "strict":
            if safety.level == "safe":
                return True
            if safety.level in ("caution", "dangerous"):
                return self._ask_user(tool_name)

        # default: 安全自动，危险询问
        if safety.level == "safe":
            return True
        if safety.level in ("caution", "dangerous"):
            return self._ask_user(tool_name)

        return True

    def _ask_user(self, tool_name: str, args: dict | None = None) -> bool:
        """询问用户是否允许调用。"""
        print(f"\n⚠️  工具调用需要确认: {tool_name}")
        if args:
            print(f"   参数: {json.dumps(args, ensure_ascii=False)[:200]}")

        response = input("   允许执行? (Y/n/a=始终允许此会话): ").strip().lower()

        if response in ("a", "always"):
            self._cache.add(tool_name)
            return True
        if response in ("", "y", "yes"):
            self._cache.add(tool_name)
            return True
        return False
```

### 7.3 权限配置

```json
{
  "permissions": {
    "policy": "default",
    "always_allow": ["read_file"],
    "always_ask": ["write_file", "run_python"],
    "mcp_servers": {
      "github": {
        "policy": "always_ask",
        "allowed_tools": ["list_issues", "get_issue"]
      }
    }
  }
}
```

### 7.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 权限模型 | **三层策略 + 二级安全等级** | 足够灵活但不复杂 |
| 会话缓存 | **默认启用，可关闭** | 用户体验好，安全可配置 |
| 询问交互 | **stdin 交互** | 保持 CLI 原生，不依赖 GUI |
| 与 MCP 权限整合 | **统一 PermissionManager** | 单一职责 |

---

## 八、上下文管理

上下文管理已有详细设计（`context_design.md`），本节聚焦 Harness 视角的关键设计。

### 8.1 Harness 中的上下文流程

```
Agent.run("分析项目")
    │
    ▼
1. ContextBuilder.build_context(user_input, session_history)
   │
   ▼                                  ┌────────────────┐
2. 组装 System Prompt (L1~L4) ──────→│  Token 估算    │
   │                                  └────────────────┘
   ▼
3. 处理会话历史 (L5):
   ├── 如果 token > 70% → 摘要旧轮次
   ├── 如果 token > 90% → 强制截断
   └── 如果 token > 95% → 拒绝（上下文溢出）
   │
   ▼
4. 构建 messages = [system, summary?, ...history, user]
   │
   ▼
5. LLM.generate(messages, tools)
   │
   ▼
6. 更新 Token 统计
   │
   ▼
7. 工具调用 → 执行 → 截断 → 注入 → 回到步骤 2
```

### 8.2 Token 预算的集成

```python
def _update_budget(self, token_usage: TokenUsage) -> None:
    """更新 Token 预算（每次 LLM 调用后）。"""
    self.budget.tokens_prompt += token_usage.prompt_tokens
    self.budget.tokens_completion += token_usage.completion_tokens
    
    # 如果接近上限，触发上下文压缩
    ratio = self.budget.tokens_prompt / self.budget.tokens_limit
    if ratio > 0.7:
        self._compress_context()
```

### 8.3 上下文压缩策略

```python
class ContextCompressor:
    """上下文压缩器：在 Token 预算紧张时压缩上下文。"""

    def __init__(
        self,
        summarizer: HistorySummarizer,
        truncator: ToolOutputTruncator,
        sliding_window: SlidingWindowStrategy,
    ):
        self.summarizer = summarizer
        self.truncator = truncator
        self.sliding_window = sliding_window

    def compress(
        self,
        messages: list[Message],
        token_budget: TokenBudget,
    ) -> list[Message]:
        """根据预算压力选择压缩策略。"""
        ratio = token_budget.usage_ratio

        if ratio < 0.5:
            return messages  # 无需压缩

        if ratio < 0.7:
            # 轻度：只截断工具输出
            return self.truncator.truncate_messages(messages)

        if ratio < 0.9:
            # 中度：摘要旧轮次
            return self.sliding_window.apply(
                messages, token_budget.limit, token_budget.used_prompt
            )

        # 重度：仅保留最后一轮
        return self.sliding_window.apply_aggressive(messages)
```

### 8.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Token 估算 | **启发式估算 + token_margin** | 零依赖，已在设计中 |
| 压缩时机 | **每次 LLM 调用后检查** | 及时，不阻塞后续 |
| 与现有 ContextBuilder 的关系 | **复用现有实现** | 避免重复设计 |

---

## 九、扩展系统

### 9.1 三层次扩展体系

```
Kocor 扩展体系:

1️⃣ Skills（技能）
   用途: 预设的提示模板或代码函数
   触发: /command 或 LLM 调用
   典型: /summarize, /uuid

2️⃣ MCP（外部工具服务器）
   用途: 通过 MCP 协议桥接外部工具
   触发: Agent 自动选择
   典型: GitHub, 数据库, 浏览器

3️⃣ Hooks（生命周期钩子）
   用途: 在 Harness 事件点注入自定义逻辑
   触发: pre/post tool, pre/post generate
   典型: 审计日志, 速率监控, 自定义权限
```

### 9.2 Hooks 系统设计

这是当前 Kocor Agent 完全缺失的机制：

```python
# ── Hook 类型 ──

class HookPoint(Enum):
    PRE_GENERATE = "pre_generate"      # LLM 调用前
    POST_GENERATE = "post_generate"    # LLM 调用后
    PRE_TOOL = "pre_tool"              # 工具执行前
    POST_TOOL = "post_tool"            # 工具执行后
    PRE_ITERATION = "pre_iteration"    # 迭代开始前
    POST_ITERATION = "post_iteration"  # 迭代结束后
    ON_ERROR = "on_error"              # 发生错误
    ON_BUDGET_EXHAUSTED = "on_budget_exhausted"  # 预算耗尽


# ── Hook 接口 ──

class Hook(Protocol):
    """Hook 接口。"""
    
    @property
    def hook_point(self) -> HookPoint:
        """此 Hook 挂载的生命周期点。"""
        ...

    def run(self, context: HookContext) -> HookResult:
        """执行 Hook 逻辑。
        
        Args:
            context: Hook 执行上下文（含当前迭代、消息、工具等）
            
        Returns:
            HookResult:
            - action: "continue" | "skip_tool" | "abort"
            - message: 附加消息
        """
        ...


# ── Hook Runner ──

class HookRunner:
    """Hook 执行器：管理所有已注册 Hook 的执行。"""

    def __init__(self):
        self._hooks: dict[HookPoint, list[Hook]] = {}

    def register(self, hook: Hook) -> None:
        """注册一个 Hook。"""
        point = hook.hook_point
        if point not in self._hooks:
            self._hooks[point] = []
        self._hooks[point].append(hook)

    def run(self, point: HookPoint, context: HookContext) -> list[HookResult]:
        """执行指定点的所有 Hook。"""
        results = []
        for hook in self._hooks.get(point, []):
            try:
                result = hook.run(context)
                results.append(result)
                if result.action == "abort":
                    break  # abort 终止后续 Hook
            except Exception as e:
                results.append(HookResult(
                    action="continue",
                    message=f"Hook error: {e}",
                ))
        return results


# ── HookContext & HookResult ──

@dataclass
class HookContext:
    """Hook 执行上下文。"""
    iteration: int
    messages: list[Message]
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    error: Exception | None = None
    config: dict = field(default_factory=dict)


@dataclass
class HookResult:
    """Hook 执行结果。"""
    action: str = "continue"  # continue | skip_tool | abort
    message: str = ""


# ── 内置 Hook 示例：审计日志 ──

class AuditLogHook:
    """审计日志 Hook：记录所有工具调用到文件。"""
    
    hook_point = HookPoint.POST_TOOL
    
    def __init__(self, log_path: str = "harness.log"):
        self.log_path = log_path
    
    def run(self, context: HookContext) -> HookResult:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": context.iteration,
            "tool": context.tool_call.function.name,
            "duration_ms": 0,  # 由调用方填充
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return HookResult(action="continue")
```

### 9.3 Hooks 配置

```json
{
  "hooks": {
    "pre_tool": [
      {
        "type": "rate_limiter",
        "config": {"max_calls_per_minute": 10}
      }
    ],
    "post_tool": [
      {
        "type": "audit_log",
        "config": {"log_path": "harness.log"}
      }
    ],
    "on_error": [
      {
        "type": "notify",
        "config": {"webhook_url": ""}
      }
    ]
  }
}
```

### 9.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Hook 数量 | **6 个关键点** | 足够覆盖大多数场景 |
| Hook 注册 | **代码注册 + JSON 配置** | 灵活且可配置 |
| 执行模型 | **同步串行** | 简单可靠，无并发问题 |
| 失败处理 | **记录错误，不中断主流程** | 稳健性优先 |

---

## 十、沙盒与安全

### 10.1 安全分层模型

```
Layer 0: Harness 层 (代码本身)
  └── 权限系统、Budget 限制、Hooks 审计

Layer 1: LLM 层
  └── System Prompt 安全约束、输出过滤

Layer 2: 工具层
  └── 文件读写范围限制、沙盒执行

Layer 3: 扩展层
  └── MCP 环境变量过滤、Skill 编码安全

Layer 4: 外部层
  └── 子进程超时、资源限制
```

### 10.2 执行沙盒增强

当前 `run_python` 只做了超时和环境变量过滤。增强方向：

```python
class Sandbox:
    """代码执行沙盒。"""

    def __init__(
        self,
        timeout: int = 30,
        memory_limit: str = "256m",     # RLIMIT_AS
        allowed_modules: set[str] | None = None,
        blocked_modules: set[str] | None = None,
        network_access: bool = False,
    ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = allowed_modules or {
            "math", "json", "re", "datetime",
            "collections", "itertools", "functools",
            "string", "typing", "pathlib", "decimal",
            "uuid", "hashlib", "statistics",
        }
        self.blocked_modules = blocked_modules or {
            "os", "subprocess", "sys", "shutil",
            "socket", "urllib", "requests", "http",
            "importlib", "ctypes", "multiprocessing",
        }
        self.network_access = network_access

    def run(self, code: str) -> SandboxResult:
        """在沙盒中执行代码。"""
        # 1. Prep: 注入安全拦截
        wrapper = self._build_wrapper(code)
        
        # 2. 子进程执行
        try:
            result = subprocess.run(
                ["python", "-c", wrapper],
                capture_output=True, text=True,
                timeout=self.timeout,
                env=self._build_env(),
            )
            return SandboxResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="", stderr="", exit_code=-1, timed_out=True
            )

    def _build_wrapper(self, code: str) -> str:
        """构建安全包装代码。"""
        # 注入 __import__ hook 拦截危险模块
        blockers = []
        for mod in self.blocked_modules:
            blockers.append(f"'{mod}'")
        
        wrapper = f"""
import builtins as __builtins__
_original_import = __builtins__.__import__

def _safe_import(name, *args, **kwargs):
    blocked = {{{','.join(blockers)}}}
    if name.split('.')[0] in blocked:
        raise ImportError(f"Module '{{name}}' is blocked for security reasons")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = _safe_import

# 用户代码
{code}
"""
        return wrapper

    def _build_env(self) -> dict[str, str]:
        """构建沙盒环境变量（移除敏感信息）。"""
        env = os.environ.copy()
        # 移除 API keys 和 tokens
        for key in list(env.keys()):
            if any(s in key.upper() for s in ("API_KEY", "SECRET", "TOKEN", "PASSWORD")):
                del env[key]
        if not self.network_access:
            # 移除代理设置
            for key in list(env.keys()):
                if key.lower().startswith("http_"):
                    del env[key]
        return env
```

### 10.3 文件 I/O 安全

```python
class FileAccessGuard:
    """文件访问守卫：限制工具的文件读写范围。"""

    def __init__(self, allowed_dir: str = ""):
        self.allowed_dir = os.path.abspath(allowed_dir) if allowed_dir else ""

    def check_read(self, path: str) -> str:
        """检查读取权限，返回规范化路径。"""
        abs_path = os.path.abspath(path)
        if self.allowed_dir and not abs_path.startswith(self.allowed_dir):
            raise PermissionError(
                f"读取 '{path}' 被拒绝：不在允许的目录内"
            )
        return abs_path

    def check_write(self, path: str) -> str:
        """检查写入权限。"""
        abs_path = os.path.abspath(path)
        if self.allowed_dir and not abs_path.startswith(self.allowed_dir):
            raise PermissionError(
                f"写入 '{path}' 被拒绝：不在允许的目录内"
            )
        # 检查 .env / .env.* 等敏感文件
        filename = os.path.basename(abs_path)
        if filename.startswith(".env"):
            raise PermissionError(f"写入 '{filename}' 被拒绝：敏感文件")
        return abs_path
```

### 10.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 沙盒技术 | **subprocess + 模块拦截** | 轻量，零额外依赖 |
| 模块白名单 vs 黑名单 | **黑名单 + 可扩展白名单** | 灵活，默认安全 |
| 文件安全 | **目录限制 + 敏感文件保护** | 防止误操作 |
| 网络限制 | **默认关闭，沙盒内无网络** | 最小权限原则 |

---

## 十一、可观测性

### 11.1 观测层次

```
可观测性金字塔：

用户可见层
  ├── 流式输出 (实时显示 Agent 思考/调用)
  ├── 进度指示 (迭代计数、耗时)
  └── 总结报告 (工具调用摘要)

调试层
  ├── 详细日志 (每次 LLM 调用的完整消息)
  ├── Token 审计 (每个 provider 调用的精确 Token 数)
  └── 工具审计 (参数、结果、耗时)

分析层
  ├── 会话回放 (消息序列)
  ├── 性能指标 (平均迭代时间、工具延迟)
  └── TOOD 调用统计 (调用频率、失败率)
```

### 11.2 日志系统

```python
# 使用标准库 logging + 结构化字段

class HarnessLogger:
    """Harness 结构化日志。"""

    def __init__(self, level: str = "INFO"):
        self.logger = logging.getLogger("kocor.harness")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def log_iteration(self, iteration: int, token_count: int) -> None:
        self.logger.info("iteration_done", extra={
            "iteration": iteration,
            "tokens": token_count,
        })

    def log_tool_call(
        self, name: str, duration_ms: float, success: bool
    ) -> None:
        self.logger.info("tool_call", extra={
            "tool": name,
            "duration_ms": duration_ms,
            "success": success,
        })

    def log_budget_warning(self, ratio: float) -> None:
        self.logger.warning("budget_warning", extra={
            "usage_ratio": ratio,
        })

    def log_error(self, component: str, error: str) -> None:
        self.logger.error("error", extra={
            "component": component,
            "error": error,
        })
```

### 11.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 日志框架 | **标准库 `logging`** | 零依赖 |
| 审计记录 | **ToolCallRecord 列表** | 轻量，内存中 |
| 用户可见信息 | **迭代计数 + 工具调用摘要** | 不多不少 |

---

## 十二、错误处理与恢复

### 12.1 错误分类

```
Agent Loop 错误类型:
├── Recoverable（可恢复）
│   ├── Tool 执行异常（超时、错误参数）
│   ├── LLM 临时错误（API 限流、网络闪断）
│   └── 权限拒绝（用户选择拒绝）
│
├── Degrade（可降级）
│   ├── Context 溢出（压缩后仍超限）
│   ├── Budget 耗尽（迭代数/Token/超时）
│   └── 部分工具不可用（MCP 离线）
│
└── Fatal（不可恢复）
    ├── 配置错误（provider/key 无效）
    ├── LLM API 永久错误（认证失败）
    └── 安全违规（目录越界）
```

### 12.2 策略

```python
class ErrorHandler:
    """Harness 错误处理策略。"""

    RETRYABLE_ERRORS = {
        "RateLimitError",
        "Timeout",
        "ServiceUnavailableError",
        "InternalServerError",
    }

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def handle_tool_error(
        self, error: Exception, tool_name: str, iteration: int
    ) -> str:
        """处理工具执行错误。"""
        error_type = type(error).__name__

        if error_type in self.RETRYABLE_ERRORS:
            return (
                f"[重试] 工具 {tool_name} 遇到临时错误 ({error_type})，"
                f"请稍后重试"
            )

        if isinstance(error, PermissionError):
            return str(error)

        # 其他错误：返回错误信息，让 LLM 决定下一步
        return f"Error executing {tool_name}: {error_type}: {error}"

    def handle_llm_error(
        self, error: Exception, retry_count: int
    ) -> tuple[bool, str]:
        """处理 LLM 调用错误。

        Returns:
            (should_retry, error_message)
        """
        error_type = type(error).__name__

        if error_type in self.RETRYABLE_ERRORS and retry_count < self.max_retries:
            wait = 2 ** retry_count
            return (True, f"LLM 临时错误，{wait}s 后重试...")

        return (False, f"LLM 错误: {error}")
```

### 12.3 优雅降级

```python
class GracefulDegradation:
    """优雅降级策略。"""

    def degrade_tools(self, error: Exception) -> None:
        """当 LLM API 离线时，工具仍可工作。"""
        # TODO: 实现离线模式
        pass

    def partial_result(self, tool_history: list[ToolCallRecord]) -> str:
        """预算耗尽时返回已有的部分结果。"""
        if not tool_history:
            return "Agent 在完成任何操作前已达到限制。"

        lines = ["Agent 已达到执行限制。已完成的操作为："]
        for rec in tool_history:
            lines.append(f"  {rec.iteration}. {rec.tool_name}()")
        return "\n".join(lines)
```

### 12.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 重试策略 | **指数退避，最大 3 次** | 常见 API 限流模式 |
| 降级策略 | **返回部分结果** | 不丢失已完成工作 |
| 错误消息 | **结构化，含 error_type** | LLM 可据此决定下一步 |

---

## 十三、配置系统

### 13.1 配置分层

```
配置优先级（从高到低）：

1. CLI 参数         --dangerous, --stream
2. 环境变量         KOCOR_*
3. 本地配置         kocor.harness.json (项目级)
4. 全局默认         代码中的 Config 默认值
```

### 13.2 Harness 配置

```python
@dataclass
class HarnessConfig:
    """Harness 专属配置（新增）。"""
    
    # 循环控制
    max_iterations: int = 20
    max_tokens_per_response: int = 4096
    max_total_time: int = 300  # 秒
    
    # 权限
    permission_policy: str = "default"  # permissive | default | strict
    permission_cache: bool = True
    
    # 上下文
    context_max_tokens: int = 200_000
    context_summary_threshold: float = 0.70
    context_truncate_threshold: float = 0.90
    preserve_rounds: int = 3
    
    # 沙盒
    sandbox_timeout: int = 30
    sandbox_memory_limit: str = "256m"
    sandbox_blocked_modules: list[str] | None = None
    sandbox_network: bool = False
    
    # Tools
    allowed_dir: str = ""  # 文件工具允许的目录
    
    # 重试
    max_retries: int = 3
    retry_delay_base: float = 1.0
```

### 13.3 配置文件

```json
{
  "harness": {
    "max_iterations": 20,
    "permission_policy": "default",
    "context_max_tokens": 200000,
    "sandbox_timeout": 30
  },
  "permissions": {
    "always_allow": ["read_file"],
    "always_ask": ["write_file", "run_python"]
  },
  "hooks": {
    "post_tool": [
      {
        "type": "audit_log",
        "config": { "log_path": ".kocor/harness.log" }
      }
    ]
  }
}
```

### 13.4 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 配置格式 | **JSON** | 与 MCP/Skills 配置文件一致 |
| 配置来源 | **环境变量 + JSON 文件** | 简单且实用 |
| 与现有 Config 的关系 | **扩展 Config** | 合并到单一配置对象 |

---

## 十四、CLI 与交互

### 14.1 CLI 增强

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Kocor Agent Harness")
    
    # 模式
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    # 权限
    parser.add_argument("--dangerous", action="store_true",
                        help="允许危险操作（不确认）")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式（每次工具调用都确认）")
    
    # 配置
    parser.add_argument("--config", help="Harness 配置文件路径")
    parser.add_argument("--max-iterations", type=int, help="最大迭代次数")
    
    # 输入
    parser.add_argument("user_input", nargs="*", help="用户问题")
    
    return parser.parse_args()
```

### 14.2 用户输出

```python
class UserInterface:
    """用户界面：统一管理用户可见的输出。"""

    def __init__(self, width: int = 60):
        self.width = width
        self.iteration = 0

    def start_session(self) -> None:
        """会话开始。"""
        print(f"\n{'=' * self.width}")
        print("  Kocor Agent")
        print(f"{'=' * self.width}\n")

    def start_iteration(self, n: int) -> None:
        """迭代开始。"""
        self.iteration = n
        title = f" 迭代 {n}"
        print(f"\n── {title} {'─' * max(0, self.width - len(title) - 4)}")

    def show_tool_call(self, name: str, args: dict) -> None:
        """显示工具调用。"""
        args_str = json.dumps(args, ensure_ascii=False)[:100]
        print(f"  🔧 {name}({args_str})")

    def show_tool_result(self, result: str) -> None:
        """显示工具结果摘要。"""
        preview = result[:200].replace("\n", " ")
        if len(result) > 200:
            preview += "..."
        print(f"  📊 {preview}")

    def show_thinking(self, text: str) -> None:
        """显示推理过程。"""
        print(f"  🧠 {text}")

    def show_answer(self, text: str) -> None:
        """显示最终答案。"""
        print(f"\n  💡 {text}")

    def show_summary(self, records: list[ToolCallRecord]) -> None:
        """显示会话总结。"""
        if not records:
            return
        print(f"\n{'─' * self.width}")
        print(f"  会话总结: {len(records)} 次工具调用")
        for rec in records:
            print(f"  #{rec.iteration} {rec.tool_name}() — {rec.duration_ms:.0f}ms")
```

### 14.3 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 输出格式 | **纯文本 + Emoji 图标** | 无需依赖，清晰可读 |
| 权限交互 | **stdin input()** | 简单，零额外依赖 |

---

## 十五、实施路径

### 15.1 依赖关系

```
Phase 1: Harness 核心
  AgentLoop 控制器 ← 依赖: agent.py 当前实现
    提取 Agent.run()/stream() 核心逻辑
    添加 Budget 追踪
    添加事件发射

Phase 2: 统一权限系统
  PermissionManager 扩展 ← 依赖: Phase 1
    覆盖所有工具类型
    三位一体策略
    会话缓存

Phase 3: Hooks 系统
  HookRunner + Hook 接口 ← 依赖: Phase 1
    6 个生命周期点
    执行器实现
    配置文件加载

Phase 4: 沙盒增强
  Sandbox + FileAccessGuard ← 依赖: 现有 run_python
    模块拦截
    文件安全
    环境变量过滤

Phase 5: 可观测性
Phase 6: 错误处理与降级
  ErrorHandler + GracefulDegradation ← 依赖: Phase 1-2
    重试策略
    降级策略
    部分结果返回

Phase 7: 配置与 CLI
  HarnessConfig + CLI 增强 ← 依赖: Phase 1-6
    配置文件
    CLI 参数扩展
    分层配置合并
```

### 15.2 实施时间线

```
Phase 1: AgentLoop         ─── 1-2 天 (核心)
Phase 2: 权限系统             ─── 1 天
Phase 3: Hooks 系统           ─── 1-2 天
Phase 4: 沙盒增强             ─── 0.5 天
Phase 5: 可观测性             ─── 1 天
Phase 6: 错误处理             ─── 0.5 天
Phase 7: 配置与 CLI           ─── 1 天
                               ─────────
                        总计: 6-9 天
```

### 15.3 测试策略

每个 Phase 遵循 TDD 模式：

```
Phase 1 测试:
├── 单元测试: AgentLoop 循环控制、Budget 追踪
├── 模拟测试: Mock LLM + Mock Tools 验证循环终止
└── 边界测试: Budget 耗尽、空响应、最大迭代

Phase 2 测试:
├── 单元测试: PermissionManager 三种策略分支
├── 集成测试: Permission + AgentLoop 联动
└── 边界测试: 缓存命中/未命中、拒绝场景

Phase 3 测试:
├── 单元测试: HookRunner 注册/执行/错误
├── 集成测试: Hook + AgentLoop 联动
└── 边界测试: Hook abort、Hook 异常

Phase 4-7: 类似模式 ...
```

### 15.4 文件结构变化

```
src/kocor/
├── harness/                          # 新增: Harness 工程模块
│   ├── __init__.py                   # 导出公开接口
│   ├── loop.py                       # AgentLoop 控制器
│   ├── budget.py                     # IterationBudget
│   ├── events.py                     # EventEmitter, HarnessEvent
│   ├── hooks.py                      # Hook 接口、HookRunner、内置 Hooks
│   ├── permission.py                 # PermissionManager（从 mcp/ 迁移）
│   ├── sandbox.py                    # Sandbox（从 tools/toolset/ 迁移增强）
│   ├── file_guard.py                 # FileAccessGuard
│   ├── logger.py                     # HarnessLogger
│   ├── error_handler.py             # ErrorHandler
│   └── config.py                     # HarnessConfig 加载
│
├── agent.py                          # 精简: 委托给 harness/
│
├── mcp/permission.py                 # 弃用: 迁移到 harness/permission.py
│
├── tools/toolset/run_python.py       # 增强: 使用 harness.sandbox
│
└── config.py                         # 扩展: 合并 HarnessConfig
```

---

## 附录 A：与竞品 Harness 对比

| Harness 能力 | Claude Code | Cline | Aider | Hermes | Kocor(当前) | Kocor(目标) |
|-------------|------------|-------|-------|--------|------------|-------------|
| 三层权限策略 | ✅ | ✅ | ⚠️ 基础 | ❌ | ⚠️ MCP 仅 | ✅ |
| Hooks 系统 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| MCP 集成 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Skills 系统 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Token 预算管理 | ✅ (隐式) | ✅ (显式) | ✅ (显式) | ❌ | ⚠️ 设计中 | ✅ |
| 历史摘要 | ⚠️ 部分 | ✅ | ❌ | ❌ | ⚠️ 设计中 | ✅ |
| 滑动窗口 | ❌ | ✅ | ✅ | ❌ | ⚠️ 设计中 | ✅ |
| 沙盒执行 | ✅ | ✅ | ✅ | ❌ | ⚠️ 基础 | ✅ |
| 调试模式 | ✅ | ⚠️ 部分 | ✅ | ❌ | ❌ | ✅ |
| 审计日志 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 优雅降级 | ✅ | ⚠️ 部分 | ❌ | ❌ | ❌ | ✅ |
| 配置分层 | ✅ | ⚠️ 单层 | ⚠️ 单层 | ❌ | ⚠️ 单层 | ✅ |

## 附录 B：关键接口速查

```python
# ── 用户主要接口 ──
agent = Agent(llm, tools, ...)
agent.run("question")          # → str（最终答案）
agent.stream("question")       # → Iterator[StreamChunk]

# ── Harness 配置接口 ──
config = load_harness_config()  # 从 JSON + ENV 加载

# ── 自定义 Hook ──
class MyHook:
    hook_point = HookPoint.POST_TOOL
    def run(self, ctx: HookContext) -> HookResult:
        ...
        
hook_runner = HookRunner()
hook_runner.register(MyHook())

# ── 工具注册 ──
registry = ToolRegistry()
registry.register("my_tool", "Does X", params, handler)

# ── 权限配置 ──
perm = PermissionManager(
    policy="default",
    always_allow={"read_file"},
    always_ask={"write_file"},
)
```

---

> **文档版本**: v0.1  
> **最后更新**: 2026-06-21  
> **关联文档**: [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) — 整体技术方案 | [context_design.md](context_design.md) — 上下文管理设计 | [streaming_design.md](streaming_design.md) — 流式输出设计
