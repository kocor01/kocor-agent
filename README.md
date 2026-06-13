# Kocor Agent

一个基于 LLM 的小而美的自主 Agent 助手。

通过调用工具（读文件、写文件、执行 Python 代码）自主完成任务，支持 OpenAI 和 Anthropic 两种 LLM 后端。

## 特性

- **自主循环**：理解意图 → 调用工具 → 观察结果 → 迭代直至完成
- **双后端**：同时支持 OpenAI 和 Anthropic 格式，通过 `KOCOR_PROVIDER` 一键切换
- **内置工具**：文件读写、沙盒执行 Python 代码
- **极简设计**：零框架依赖，只依赖 `openai`、`anthropic`、`python-dotenv`
- **CLI 友好**：命令行参数或管道输入，开箱即用

## 安装

```bash
git clone https://github.com/kocor01/kocor-agent
cd kocor_agent
uv venv --python 3.12
uv pip install -e .
```

## 配置

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
# 选择后端：openai 或 anthropic
KOCOR_PROVIDER=openai

# OpenAI 配置
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=http://0.0.0.0:8081  # 兼容端点（可选）

# Anthropic 配置
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_BASE_URL=http://0.0.0.0:8081  # 兼容端点（可选）
```

`KOCOR_PROVIDER` 支持大小写混写（`openai` / `OpenAI` / `anthropic` / `Anthropic`）。

## 使用

### 命令行

```bash
# 直接传入问题
python -m kocor "帮我读取 .env 的内容"

# 管道输入
echo "统计当前目录下 .py 文件数量" | python -m kocor
```

### 作为库使用

```python
from kocor.agent import Agent
from kocor.config import load_config
from kocor.llm_client import create_llm_client
from kocor.tools import create_default_tools

config = load_config()
llm = create_llm_client(config)
tools = create_default_tools(config)
agent = Agent(llm=llm, tools=tools)

result = agent.run("帮我读取 .env 的内容")
print(result)
```

### 自定义系统提示词

```python
agent = Agent(
    llm=llm,
    tools=tools,
    system_prompt="你是一个代码审查助手...",
    max_iterations=10,
)
```

## 架构

```
kocor/
├── __init__.py          # 包入口，版本信息
├── __main__.py          # CLI 入口
├── agent.py             # Agent 核心循环
├── config.py            # 配置加载
├── message.py           # 统一消息数据模型
├── llm_client.py        # LLM 客户端 Protocol 接口 + 工厂
├── openai_client.py     # OpenAI 兼容客户端
├── anthropic_client.py  # Anthropic 兼容客户端
└── tools.py             # 工具注册表与内置工具
```

### 核心设计

- **Protocol 接口**：`LLMClient` 定义标准接口，各 provider 实现格式转换
- **统一消息格式**：内部使用 `Message` dataclass，屏蔽 provider 差异
- **装饰器式工具注册**：`ToolRegistry` 管理工具定义与执行
- **环境变量配置**：所有配置通过 `.env` 管理，支持默认值

## 开发

```bash
# 创建虚拟环境
uv venv --clear --python 3.12

# 安装开发依赖（无额外依赖，核心包已包含全部运行时依赖）
uv pip install -e .

# 运行测试（TDD）
uv run pytest tests/ -v

# 运行所有测试
uv run pytest tests/
```

当前测试覆盖：消息模型、配置加载、工具注册与执行、OpenAI/Anthropic 客户端、Agent 循环。

## 环境变量参考

| 变量 | 说明 | 默认值 |
|---|---|---|
| `KOCOR_PROVIDER` | 后端选择（openai / anthropic） | `openai` |
| `OPENAI_API_KEY` | OpenAI API Key | 必填 |
| `OPENAI_MODEL` | OpenAI 模型名 | `gpt-4o` |
| `OPENAI_BASE_URL` | 兼容端点（可选） | — |
| `ANTHROPIC_API_KEY` | Anthropic API Key | 必填 |
| `ANTHROPIC_MODEL` | Anthropic 模型名 | `claude-sonnet-4-20250514` |
| `ANTHROPIC_BASE_URL` | 兼容端点（可选） | — |
| `KOCOR_MAX_ITERATIONS` | Agent 最大迭代次数 | `20` |
| `KOCOR_TIMEOUT` | 工具执行超时（秒） | `30` |

## License

MIT
