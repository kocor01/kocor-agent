# Kocor Agent

一个基于 LLM 的轻量自主 Agent 助手。

通过 ReAct 循环（推理→工具调用→观察→迭代）自主完成复杂任务，支持 OpenAI 和 Anthropic 双后端。

## 特性

- **ReAct 自主循环** — 推理 → 调用工具 → 观察结果 → 迭代直至完成
- **双 LLM 后端** — OpenAI 与 Anthropic 一键切换，统一 Protocol 接口
- **内置工具** — 安全文件读写
- **MCP 集成** — 通过 [Model Context Protocol](https://github.com/modelcontextprotocol) 接入外部服务器（文件系统、搜索等）
- **Skill 系统** — slash 命令 (`/summarize`) 或 LLM 可调用的插件化能力
- **上下文管理** — 分层系统提示词（身份 + 项目指令 + 环境 + 持久记忆），支持滑动窗口与激进压缩策略
- **三级权限管控** — `permissive` / `default` / `strict`，危险操作需确认
- **文件访问防护** — 路径穿越防护与敏感文件（`.env` 等）读取拦截
- **事件与 Hook 系统** — 生命周期事件发布订阅 + 可中断的拦截器链
- **流式输出** — 实时展示推理过程、回答与工具调用
- **持久记忆** — 基于文件的长期记忆，跨会话保持上下文
- **重复工具调用检测** — 自动识别并终止 3 次以上重复工具调用的死循环
- **极简依赖** — 仅依赖 `openai`、`anthropic`、`python-dotenv`、`mcp`、`requests`

## 安装

```bash
git clone https://github.com/kocor01/kocor-agent
cd kocor_agent
uv venv --python 3.12
uv pip install -e .
```

## 配置

复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
```

关键配置：

```bash
# 后端选择：openai 或 anthropic
KOCOR_PROVIDER=openai

# API Key 与模型
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o

# 或
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

完整环境变量参考见[下文](#环境变量参考)。

## 使用

### CLI

```bash
# 直接传入问题
python -m kocor "读取 .env 的内容"

# 流式输出（展示推理与工具调用的实时过程）
python -m kocor --stream "分析项目结构"

# 管道输入
echo "统计当前目录下 .py 文件数量" | python -m kocor

# REPL 交互模式
python -m kocor --repl
```

### 权限模式

```bash
# 宽松模式 — 危险操作自动允许
python -m kocor --permissive "修改文件"

# 严格模式 — 每次工具调用需确认
python -m kocor --strict "读取配置文件"
```

### 作为库使用

```python
from kocor.agent import Agent
from kocor.config import Config
from kocor.llm_provider.llm_factory import LlmFactory
from kocor.tools.tool_manager import ToolManager
from kocor.tools.permission import PermissionManager
from kocor.harness import IterationBudget

Config.load()

tool_manager = ToolManager()
tool_manager.register_all()

agent = Agent(
    llm=LlmFactory.create(),
    tool_manager=tool_manager,
    permission_mgr=PermissionManager(policy="permissive", tool_manager=tool_manager),
    budget=IterationBudget(max_iterations=20),
)

result = agent.run("分析当前项目结构")
print(result)
```

## 架构

```
src/kocor/
├── __init__.py              # 包入口
├── __main__.py              # CLI 入口（argparse、REPL、管道输入）
├── config.py                # Config 单例 — 从 .env 加载配置
├── agent.py                 # Agent 门面 — slash 命令路由，委托 Loop 引擎
├── loop.py                  # ReAct 循环引擎 — 生成 → 工具执行 → 迭代
│
├── llm_provider/            # LLM 抽象层
│   ├── llm_client.py        # LLMClient Protocol（generate / stream）
│   ├── llm_factory.py       # LlmFactory — 纯工厂，每次 create() 返回新实例
│   ├── message.py           # 统一数据模型：Message, ToolCall, StreamChunk
│   └── providers/
│       ├── openai_client.py     # OpenAI SDK 集成
│       └── anthropic_client.py  # Anthropic SDK 集成
│
├── tools/                   # 工具系统
│   ├── tool_manager.py      # ToolManager — 注册、执行、MCP/Skill 集成
│   ├── definitions.py       # ToolDefinition 数据模型
│   ├── permission.py        # PermissionManager — 三级权限策略
│   ├── truncate.py          # ToolOutputTruncator — 三级输出截断
│   ├── tool_utils.py        # 路径安全与环境变量脱敏
│   └── toolset/             # 内置工具
│       ├── read_file.py     # read_file（安全）
│       └── write_file.py    # write_file（危险）
│
├── context/                 # 上下文管理系统
│   ├── context_manager.py   # ContextManager — 消息组装与历史管理
│   ├── system_prompt.py     # SystemPromptBuilder — 四层提示词组装
│   ├── strategies.py        # ContextStrategyApplier — 调度压缩策略
│   ├── sliding_window.py    # SlidingWindowStrategy — 保留首尾 + 摘要压缩
│   ├── summarizer.py        # HistorySummarizer — LLM 历史摘要
│   ├── memory.py            # MemoryManager — 文件持久记忆
│   ├── token_counter.py     # TokenCounter — 启发式估算
│   ├── budget.py            # TokenBudget — 上下文窗口跟踪
│   └── types.py             # ContextStrategy 枚举与数据类型
│
├── harness/                 # 运行时系统
│   ├── budget.py            # IterationBudget — ReAct 循环迭代预算
│   ├── sandbox.py           # Sandbox — 子进程代码执行
│   ├── logger.py            # Logger — 按日轮转文件日志
│   └── event/               # 事件系统
│       ├── event_manager.py     # EventEmitter + Event
│       ├── event_subscribe.py   # EventSubscribe — 注册标准订阅
│       └── subscribes/logs.py   # 事件写入日志
│
├── hook/                    # Hook 系统（生命周期拦截器）
│   ├── base.py              # HookPoint, HookContext, Hook Protocol
│   ├── hook_manager.py      # HookManager — 注册/执行
│   └── hooks/audit_log.py   # AuditLogHook — 工具调用审计日志
│
├── mcp/                     # MCP 集成
│   ├── config.py            # MCPConfig + 加载服务器配置
│   ├── client.py            # MCPClient — MCP SDK 同步封装
│   ├── event_loop.py        # 后台 asyncio 事件循环桥接
│   └── mcp_manager.py       # McpManager — 连接 MCP 服务器，注册工具
│
└── skill/                   # Skill 系统
    ├── types.py             # SkillDefinition, SkillType, SkillResult
    ├── skill_manager.py     # SkillManager — 加载、发现、执行
    └── script/uuid_gen/     # 示例 CODE Skill
```

### 核心流程

1. **CLI 启动** — `__main__.py` 解析参数，加载配置，组装 `ToolManager`（内置工具 → MCP 工具 → Skill 工具）、`PermissionManager`、`HookManager`、`EventEmitter`、`IterationBudget`，创建 `Agent`
2. **Agent 分发** — `agent.py` 检测 slash 命令则路由到 `SkillManager`，否则委托 `Loop` 引擎
3. **ReAct 循环** — `loop.py` 循环执行：压缩上下文 → LLM 生成 → 追加响应 → 检查工具调用 → 权限校验 → 执行工具 → 截断输出 → 追加结果 → 重复检测 → 下一轮
4. **上下文管理** — 超阈值时自动切换滑动窗口或激进压缩策略，LLM 生成历史摘要
5. **资源清理** — 任务完成或预算耗尽后关闭 MCP 连接，解压会话历史

## 环境变量参考

| 变量 | 说明 | 默认值 |
|---|---|---|
| `KOCOR_PROVIDER` | 后端选择（`openai` / `anthropic`） | `openai` |
| `OPENAI_API_KEY` | OpenAI API Key | — |
| `OPENAI_MODEL` | OpenAI 模型名 | `gpt-4o` |
| `OPENAI_BASE_URL` | OpenAI 兼容端点（可选） | — |
| `ANTHROPIC_API_KEY` | Anthropic API Key | — |
| `ANTHROPIC_MODEL` | Anthropic 模型名 | `claude-sonnet-4-20250514` |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容端点（可选） | — |
| `KOCOR_PERMISSION_POLICY` | 权限策略（`default` / `strict` / `permissive`） | `default` |
| `KOCOR_MAX_ITERATIONS` | Agent 最大迭代次数 | `20` |
| `KOCOR_TOOL_TIMEOUT` | 工具执行超时（秒） | `30` |
| `KOCOR_MCP_CONFIG` | MCP 服务器配置文件路径 | `kocor.mcp.json` |
| `KOCOR_SKILLS_CONFIG` | Skill 配置文件路径 | `kocor.skills.json` |
| `KOCOR_MEMORY_DIR` | 上下文记忆目录 | — |
| `KOCOR_PROJECT_INSTRUCTIONS_PATH` | 项目指令文件路径 | `KOCOR.md` |
| `KOCOR_CONTEXT_STRATEGY` | 上下文策略（`default` / `sliding` / `aggressive`） | `default` |
| `KOCOR_CONTEXT_MAX_TOKENS` | 上下文窗口上限 | `200000` |
| `KOCOR_CONTEXT_SUMMARY_THRESHOLD` | 摘要触发阈值（0~1） | `0.70` |
| `KOCOR_CONTEXT_TRUNCATE_THRESHOLD` | 截断触发阈值（0~1） | `0.90` |
| `KOCOR_PRESERVE_ROUNDS` | 滑动窗口保留完整轮次数 | `3` |

## 扩展

### MCP 服务器

在 `kocor.mcp.json` 中定义外部工具服务器：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

每工具可单独配置权限覆盖。

### Skill 系统

支持两种 Skill 类型：

- **PROMPT** — 模板字符串，在调用时渲染
- **CODE** — Python 函数，直接执行

通过 `kocor.skills.json` 配置或 `skill/script/` 目录自动发现。

## 开发

```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
uv run pytest tests/ -v

# 类型检查
uv run mypy src/kocor/

# 代码检查
uv run ruff check src/kocor/
```

本项目严格遵循 TDD（测试驱动开发）：先写失败测试（Red）→ 最小代码通过（Green）→ 重构（Refactor）。

## License

MIT