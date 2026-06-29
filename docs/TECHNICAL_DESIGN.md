# Kocor Agent 技术方案

> 小而美的 LLM 自主 Agent 助手，不是通用 Agent 框架。

---

## 一、项目定位

| 维度 | 说明 |
|------|------|
| **是什么** | 一个极简的 LLM 自主 Agent，能理解意图、调用工具、完成任务 |
| **不是什么** | 不是 LangChain / AutoGen 那样的通用框架 |
| **核心能力** | 对话 + 工具调用（读文件、写文件、沙盒执行 Python） |
| **目标用户** | 开发者个人助手，嵌入工作流 |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────┐
│                    CLI / API                     │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                  Agent (核心循环)                 │
│  ┌───────────────────────────────────────────┐  │
│  │  loop: query LLM → call tool → observe →  │  │
│  │          loop until final answer           │  │
│  └───────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│  LLM Client 抽象层  ──┬── OpenAI SDK            │
│                       ├── Anthropic SDK         │
│                       └── 消息格式归一化         │
├─────────────────────────────────────────────────┤
│  Tool 系统  ────────┬── read_file               │
│                     ├── write_file              │
│                     └── run_python (沙盒)       │
├─────────────────────────────────────────────────┤
│  Message 模型层  ────┬── UserMessage            │
│                      ├── AssistantMessage       │
│                      ├── ToolMessage            │
│                      └── ToolCall / ToolResult  │
└─────────────────────────────────────────────────┘
```

### 2.1 核心循环

```
初始消息 → Agent.loop()
                │
                ▼
        ┌─ 收集所有消息（含历史）
        │
        ▼
   ┌─ normalize_messages() ──→ 转为 provider 格式
   │
   ▼
   LLM.generate(messages, tools)
   │
   ▼
   检查响应: 纯文本 or 工具调用?
   │
   ├─ 纯文本 → 返回最终答案 ✓
   │
   └─ 工具调用 → 执行工具 → 追加 ToolMessage → 回到 LLM.generate
                      │
                      ▼
                 最多 MAX_ITERATIONS (默认 20) 次循环
```

**循环终止条件：**
1. LLM 返回纯文本（无工具调用）→ 成功
2. 达到 MAX_ITERATIONS 上限 → 超时，返回已有结果
3. 工具执行异常 → 返回错误信息给用户

---

## 三、数据模型

### 3.1 消息模型

内部统一格式，不依赖任何 provider：

```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str

@dataclass
class ToolCall:
    id: str                      # provider 生成的调用 ID
    type: str = "function"       # 目前只支持 function
    function: FunctionCall

@dataclass
class FunctionCall:
    name: str
    arguments: str               # JSON 字符串

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
```

**设计决策：为什么保留 `ToolCall.id`？**
- OpenAI API 要求 `tool_calls` 中的每个调用都有唯一 ID
- 响应中会返回 `tool_call_id` 用于关联结果
- 这是 API 契约，不是可选设计

### 3.2 工具注册模型

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict             # JSON Schema (type=object, properties, required)
```

所有工具统一用 JSON Schema 描述参数，OpenAI 和 Anthropic 都支持。

---

## 四、LLM Client 抽象层

### 4.1 设计原则

- **只做归一化**：不封装 LLM 的能力差异，只转换消息格式
- **SDK 优先**：使用官方 Python SDK（`openai` / `anthropic`），不自己发 HTTP
- **最小接口**：一个方法搞定

### 4.2 接口定义

```python
class LLMClient(Protocol):
    """LLM 客户端抽象接口"""

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Message:
        """
        生成响应。

        返回:
        - 纯文本: Message(role="assistant", content="...")
        - 工具调用: Message(role="assistant", content="", tool_calls=[...])
        """
        ...

    @property
    def provider(self) -> str:
        """返回当前 provider: 'openai' | 'anthropic'"""
        ...
```

### 4.3 消息格式归一化

三种格式对照：

| 内部格式 | OpenAI 格式 | Anthropic 格式 |
|---------|------------|---------------|
| `system` | `{"role":"system","content":"..."}` | `system` 顶层参数 |
| `user` | `{"role":"user","content":"..."}` | `{"role":"user","content":"..."}` |
| `assistant` (纯文本) | `{"role":"assistant","content":"..."}` | `{"role":"assistant","content":[{"type:"text","text":"..."}]}` |
| `assistant` (工具调用) | `{"role":"assistant","tool_calls":[...]}` | `{"role":"assistant","content":[{"type:"tool_use","id":"...","name":"...","input":{...}}]}` |
| `tool` | `{"role":"tool","tool_call_id":"...","content":"..."}` | `{"role":"user","content":[{"type:"tool_result","tool_use_id":"...","content":"..."}]}` |

**归一化方向：**
```
内部 Message → provider 格式 → SDK → provider 响应 → 内部 Message
```

### 4.4 Provider 选择策略

通过环境变量 `KOCOR_PROVIDER` 控制：

```python
# 支持的值
# - "openai"     → 使用 openai SDK
# - "anthropic"  → 使用 anthropic SDK
# 默认: "openai"
```

API Key 读取：
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`

模型名读取：`KOCOR_MODEL`（如 `gpt-4o` / `claude-sonnet-4-20250514`）

### 4.5 工厂函数

```python
def create_llm_client(config: Config) -> LLMClient:
    """根据配置创建对应的 LLM 客户端"""
    match config.provider:
        case "openai":
            return OpenAIClient(config)
        case "anthropic":
            return AnthropicClient(config)
        case unknown:
            raise ValueError(f"不支持的 provider: {unknown}")
```

---

## 五、Tool 系统

### 5.1 设计理念

- **装饰器注册**：简洁声明工具
- **JSON Schema 描述**：与 LLM 工具调用标准一致
- **沙盒隔离**：代码执行在 subprocess 中，有超时和资源限制

### 5.2 工具注册

```python
class ToolRegistry:
    """工具注册与执行中心"""

    def register(self, tool: ToolDefinition, handler: Callable) -> None:
        """注册工具及其处理器"""

    def get_definitions(self) -> list[ToolDefinition]:
        """返回所有工具的 JSON Schema 定义（供 LLM 使用）"""

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """根据 tool_call 执行对应的 handler，返回结果"""
```

### 5.3 内置工具

| 工具名 | 描述 | 参数 |
|-------|------|------|
| `read_file` | 读取文件内容 | `path: str` - 文件路径 |
| `write_file` | 写入文件内容 | `path: str`, `content: str` |
| `run_python` | 在沙盒中执行 Python 代码 | `code: str` - Python 代码字符串 |

### 5.4 沙盒执行设计

`run_python` 使用 `subprocess` 启动独立 Python 进程：

```python
def run_python(code: str) -> str:
    """
    在隔离的子进程中执行 Python 代码。

    安全措施:
    - 设置超时（默认 30s）
    - 不导入危险模块（os, subprocess, sys 等通过白名单控制）
    - 限制输出大小
    - 捕获 stdout/stderr
    """
```

**沙盒策略：白名单导入**

允许导入的模块：`math`, `json`, `re`, `datetime`, `collections`, `itertools`, `functools`, `string`, `typing`, `pathlib`, `decimal`, `uuid`, `hashlib`

禁止导入：`os`, `subprocess`, `sys`, `shutil`, `socket`, `urllib`, `requests`, `http` 等系统/网络模块。

实现方式：在子进程中用 `sitecustomize` 或启动时 hook `__import__` 来拦截。

> **简化方案**：初期先用 `timeout=30s` + 捕获输出 + 不限制导入（信任环境），后续再加固。

---

## 六、Agent 核心

### 6.1 Agent 类

```python
class Agent:
    """自主 Agent 核心"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
    ):
        ...

    def run(self, user_input: str) -> str:
        """
        执行一次完整的 Agent 循环。

        返回最终文本答案。
        """
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]

        for _ in range(self.max_iterations):
            # 1. 调用 LLM
            response = self.llm.generate(messages, tools=self.tools.get_definitions())
            messages.append(response)

            # 2. 检查是否有工具调用
            if not response.tool_calls:
                return response.content  # 最终答案

            # 3. 执行工具
            for tool_call in response.tool_calls:
                result = self.tools.execute(tool_call)
                messages.append(Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                ))

        # 超时
        return f"Agent 在 {self.max_iterations} 次迭代后仍未完成，可能任务过于复杂。"
```

### 6.2 System Prompt 设计

```python
DEFAULT_SYSTEM_PROMPT = """\
你是一个名为 Kocor 的 AI 助手，擅长通过调用工具来完成任务。

你的能力:
- 读取和写入文件
- 在沙盒中执行 Python 代码

工作原则:
1. 理解用户意图后，选择合适的工具完成任务
2. 如果需要多次操作，逐步执行，每次只做一个合理的操作
3. 工具执行后，根据结果决定下一步
4. 任务完成后，给出清晰简洁的总结
5. 如果不确定，可以向用户提问（通过回复纯文本）\
"""
```

---

## 七、项目结构

```
kocor_agent/
├── pyproject.toml              # 项目配置 + 依赖
├── .env.example                # 环境变量模板
├── .env                        # 本地环境变量（gitignore）
├── CLAUDE.md                   # 开发规范
├── README.md                   # 项目说明
├── docs/
│   └── TECHNICAL_DESIGN.md     # 本文档
│
├── src/
│   └── kocor/
│       ├── __init__.py         # 版本信息
│       ├── __main__.py         # 入口: python -m kocor
│       │
│       ├── message.py          # Message / ToolCall / ToolResult 数据模型
│       ├── llm_client.py       # LLMClient 抽象 + create_llm_client 工厂
│       ├── openai_client.py    # OpenAI SDK 实现
│       ├── anthropic_client.py # Anthropic SDK 实现
│       ├── tools.py            # ToolRegistry + 内置工具实现
│       ├── agent.py            # Agent 核心循环
│       └── config.py           # 配置加载（从 .env）
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # pytest fixtures
    ├── test_message.py         # 消息模型测试
    ├── test_llm_client.py      # LLM 客户端抽象测试（mock）
    ├── test_openai_client.py   # OpenAI 客户端测试（mock API）
    ├── test_anthropic_client.py# Anthropic 客户端测试（mock API）
    ├── test_tools.py           # 工具注册与执行测试
    ├── test_agent.py           # Agent 循环测试（mock LLM）
    └── test_config.py          # 配置加载测试
```

---

## 八、依赖管理

### 8.1 pyproject.toml

```toml
[project]
name = "kocor-agent"
version = "0.1.0"
description = "小而美的 LLM 自主 Agent 助手"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.0.0",
    "anthropic>=0.42.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8.0",
    "mypy>=1.11",
]

[tool.uv]
# uv 配置
```

### 8.2 依赖原则

- **最小依赖**：只引入必要的包
- SDK 本身可能已有 httpx/typing-extensions 等传递依赖，接受它们
- 不引入 pydantic、langchain 等重型框架

---

## 九、关键设计决策

### 9.1 为什么不用 pydantic？

项目定位是"小而美"，数据模型简单且固定。`dataclass` 足够表达，引入 pydantic 会增加依赖和复杂度。

### 9.2 为什么不用 async？

初期用同步接口，简单直接。如果后续需要（如 API 并发），可以加 `async` 版本。同步接口更符合"最小化"原则。

### 9.3 为什么消息用 dataclass 而不是 dict？

- 类型安全：IDE 自动补全，静态检查友好
- 意图清晰：`message.role` 比 `message["role"]` 可读性好
- 零运行时开销：dataclass 比 dict 更轻量

### 9.4 工具调用为什么保留 ToolCall.id？

OpenAI API 要求 `tool_calls` 中的每个调用有唯一 ID，响应中通过 `tool_call_id` 关联结果。这是 API 契约层面的要求，不是可选设计。

### 9.5 为什么用 subprocess 而不是 exec() 做沙盒？

- `exec()` 无法隔离文件系统、网络等系统资源
- `subprocess` 启动独立进程，天然隔离
- 可以设置超时（`timeout` 参数）
- 可以捕获输出（`stdout`/`stderr`）

---

## 十、开发顺序（TDD）

按照 TDD 模式，按以下顺序开发：

```
Phase 1: 基础设施
  1. 初始化项目结构 (pyproject.toml, .env.example)
  2. 编写 Message 模型测试 → 实现 message.py
  3. 编写 config 测试 → 实现 config.py

Phase 2: LLM 客户端
  4. 编写 LLMClient 抽象接口测试
  5. 实现 openai_client.py（mock OpenAI API）
  6. 实现 anthropic_client.py（mock Anthropic API）
  7. 实现 create_llm_client 工厂函数

Phase 3: 工具系统
  8. 编写 ToolRegistry 测试 → 实现 tools.py（骨架）
  9. 实现 read_file / write_file / run_python
  10. 为每个工具编写测试

Phase 4: Agent 核心
  11. 编写 Agent 循环测试（mock LLM 和 Tools）→ 实现 agent.py
  12. 集成测试：端到端验证完整循环

Phase 5: 入口与打磨
  13. 实现 __main__.py (CLI 入口)
  14. 编写 README.md
  15. 代码审查、重构、整理
```

---

## 十一、环境变量

```bash
# .env
KOCOR_PROVIDER=openai          # openai | anthropic
KOCOR_MODEL=gpt-4o             # 模型名
OPENAI_API_KEY=sk-xxx          # OpenAI API Key
ANTHROPIC_API_KEY=sk-ant-xxx   # Anthropic API Key
KOCOR_MAX_ITERATIONS=20        # 最大迭代次数
KOCOR_TOOL_TIMEOUT=30          # 工具执行超时（秒）
```

---

## 十二、扩展预留（不实现，只预留接口）

以下能力**本次不实现**，但架构上预留扩展点：

| 能力 | 扩展方式 |
|------|---------|
| 更多 LLM provider | 新增 `xxx_client.py`，工厂函数加分支 |
| 更多工具 | `ToolRegistry.register()` 注册即可 |
| 流式输出 | `LLMClient.stream()` 方法 |
| 多轮对话持久化 | `Agent` 接受 `messages` 列表初始化 |
| 规划能力（Plan-Do） | `Agent` 内部维护 planner 状态 |
| Web API | 在 `__main__.py` 上加 FastAPI 路由 |

---

## 十三、质量要求

| 维度 | 标准 |
|------|------|
| 测试覆盖率 | 核心逻辑 ≥ 90% |
| 类型标注 | 全部函数有类型标注 |
| 代码风格 | Ruff lint + format |
| 文档 | 每个公开类/函数有 docstring |
| Git | 每个功能一个 commit，commit message 遵循 Conventional Commits |
