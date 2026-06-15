# 流式输出技术方案

## 1. 背景与目标

### 现状
- `LLMClient.generate()` 阻塞等待完整响应后才返回 `Message`
- CLI 一次性打印最终结果，用户需等待 LLM 思考+工具执行完毕
- TECHNICAL_DESIGN.md 已预留 `LLMClient.stream()` 扩展点，尚未实现

### 目标
- 用户输入后**立即看到文字逐字输出**，降低感知延迟
- 工具调用阶段仍为阻塞（工具执行不可流式），但工具执行前的文本可流式展示
- 保持现有同步 API 不变，向后兼容

### 范围
- 仅 OpenAI 和 Anthropic 两个 provider
- 不涉及异步改造（async/await 不在本次范围内）

---

## 2. 核心设计

### 2.1 数据模型

新增 `StreamChunk` 数据类，作为流式传输的最小单位：

```python
@dataclass
class StreamChunk:
    """流式输出数据块。

    Attributes:
        content: 本轮增量文本（非累积）。首次为起始文本，后续为增量。
                 纯文本阶段每次有值；工具调用阶段为 ""。
        tool_calls: 本轮新增的工具调用（仅工具调用阶段有值）。
                    格式与 Message.tool_calls 一致。
        is_final: 是否为最后一个 chunk（本次 LLM 响应结束）。
                  True 时 content 可能为空（最后一个 token 已在前一个 chunk 发出）。
    """
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_final: bool = False
```

**设计决策：**
- `content` 用增量而非累积，下游消费更轻量（直接 append 即可，无需 diff）
- `tool_calls` 每次携带**完整列表**（OpenAI SDK chunk 的 tool_calls 即为完整快照；Anthropic 需自行累积）
- `is_final` 标记单次 LLM 响应结束（非整个 Agent 循环结束）

### 2.2 Protocol 扩展

在 `LLMClient` Protocol 中新增 `stream()` 方法：

```python
class LLMClient(Protocol):
    @property
    def provider(self) -> str: ...

    def generate(self, ...) -> Message: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Iterator[StreamChunk]:
        """流式生成响应。

        Yields:
            StreamChunk: 流式数据块
        """
        ...
```

**设计决策：**
- `generate()` 保持不变，不做重构
- `stream()` 作为新增方法，遵循 Protocol 要求所有实现必须提供
- 签名与 `generate()` 一致，参数完全对齐

### 2.3 Agent 层扩展

新增 `Agent.stream()` 方法：

```python
class Agent:
    def run(self, user_input: str) -> str: ...

    def stream(self, user_input: str) -> Iterator[StreamChunk]:
        """流式执行 Agent 循环。

        Yields:
            StreamChunk: 流式数据块，跨多轮 LLM 响应累积输出
        """
```

**Agent 循环中的流式行为：**

```
1. 调用 llm.stream()，yield 每个 chunk 给下游
2. 收到 is_final=True 的 chunk：
   a. 如果有 tool_calls → 执行工具 → 将 tool result 追加到 messages
   b. 如果没有 tool_calls → 标记 Agent 循环结束
3. 回到步骤 1，继续下一轮 LLM 调用
4. 循环结束时 yield 一个 is_final=True 的 chunk
```

**关键设计：**
- 工具执行阶段**不 yield chunk**（工具执行是阻塞的，期间不做任何输出）
- 用户感知：文字流式输出 → 短暂停顿（工具执行）→ 继续流式输出
- 最后一轮纯文本回复的 `is_final=True` chunk 标记 Agent 循环结束

### 2.4 CLI 改造

在 `__main__.py` 中支持 `--stream` 参数：

```python
def main() -> None:
    # ... 现有初始化代码 ...

    # 运行 Agent
    if hasattr(agent, 'stream'):
        for chunk in agent.stream(user_input):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    print(f"\n[工具调用] {tc.function.name}({tc.function.arguments})")
    else:
        result = agent.run(user_input)
        print(result)
```

**设计决策：**
- `--stream` 为可选参数，默认关闭（保持原有行为）
- 纯文本模式下 `print(..., end="", flush=True)` 实现逐字输出
- 工具调用用 `[工具调用]` 前缀标记，换行展示，清晰区分

---

## 3. Provider 实现细节

### 3.1 OpenAI 实现

**SDK 流式 API：**

```python
response = client.chat.completions.create(
    model=self._model,
    messages=openai_messages,
    max_tokens=max_tokens,
    temperature=temperature,
    tools=openai_tools,
    stream=True,  # 关键参数
)
# response 为 StreamingChatCompletionChunk 的迭代器
for chunk in response:
    # chunk.choices[0].delta.content  → str | None（增量文本）
    # chunk.choices[0].delta.tool_calls → list[ToolCallDelta] | None
    # chunk.choices[0].finish_reason    → str | None
```

**实现要点：**

```python
def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
    client = OpenAI(api_key=self._api_key, base_url=self._base_url)
    openai_messages = self._normalize_in(messages)
    openai_tools = [t.to_dict() for t in tools] if tools else None

    accumulated_tool_calls: dict[int, ToolCall] = {}

    for chunk in client.chat.completions.create(
        model=self._model,
        messages=openai_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=openai_tools,
        stream=True,
    ):
        delta = chunk.choices[0].delta

        # 构建 chunk
        stream_chunk = StreamChunk(
            content=delta.content or "",
            is_final=(chunk.choices[0].finish_reason is not None),
        )

        # 累积 tool_calls
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = ToolCall(
                        id=tc_delta.id or "",
                        type=tc_delta.type or "function",
                        function=FunctionCall(
                            name=tc_delta.function.name or "",
                            arguments=tc_delta.function.arguments or "",
                        ),
                    )
                else:
                    # 增量拼接 arguments
                    if tc_delta.function.arguments:
                        accumulated_tool_calls[idx].function.arguments += tc_delta.function.arguments
            stream_chunk.tool_calls = list(accumulated_tool_calls.values())

        yield stream_chunk
```

**注意事项：**
- OpenAI SDK 的 `delta.tool_calls` 中每个 tool_call 带 `index` 字段，用于区分多个工具调用
- `arguments` 是增量拼接，需要累积
- `finish_reason` 为 `"tool_calls"` 或 `"stop"` 时标记 `is_final=True`

### 3.2 Anthropic 实现

**SDK 流式 API：**

```python
response = client.messages.create(
    model=self._model,
    system=system_content or None,
    messages=anthropic_messages,
    max_tokens=max_tokens,
    temperature=temperature,
    tools=anthropic_tools,
    stream=True,
)
# response 为 StreamEvent 的迭代器
for event in response:
    # event.type → "content_block_start" | "content_block_delta" | "content_block_stop"
    #              | "message_start" | "message_delta" | "message_stop" | "ping"
```

**事件类型与处理：**

| 事件类型 | 携带信息 | 处理方式 |
|---------|---------|---------|
| `message_start` | 消息元数据 | 忽略（无内容） |
| `content_block_start` | 工具块开始（type="tool_use"） | 开始累积新 tool_call |
| `content_block_delta` | 增量（type="input_json" 或 "text"） | 追加到对应累积器 |
| `content_block_stop` | 工具块结束 | 完成当前 tool_call 累积 |
| `message_delta` | stop_reason | 标记 is_final |
| `message_stop` | 消息结束 | 忽略（message_delta 已处理） |
| `ping` | 心跳 | 忽略 |

**实现要点：**

```python
def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
    client = Anthropic(api_key=self._api_key, base_url=self._base_url)
    # ... normalize ...

    accumulated_text = ""
    accumulated_tool_calls: dict[str, ToolCall] = {}  # key=tool_use.id

    for event in client.messages.create(
        model=self._model,
        system=system_content or None,
        messages=anthropic_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=anthropic_tools,
        stream=True,
    ):
        stream_chunk = StreamChunk()

        match event.type:
            case "content_block_delta":
                if event.delta.type == "text_delta":
                    text = event.delta.text
                    accumulated_text += text
                    stream_chunk.content = text
                elif event.delta.type == "input_json_delta":
                    idx = event.index
                    json_fragment = event.delta.partial_json
                    if idx not in accumulated_tool_calls:
                        # 从 content_block_start 获取 tool_call 骨架
                        ...
                    accumulated_tool_calls[idx].function.arguments += json_fragment
                    stream_chunk.tool_calls = list(accumulated_tool_calls.values())

            case "content_block_stop":
                # 工具块结束，确保 yield 一次完整 tool_calls
                if event.index in accumulated_tool_calls:
                    stream_chunk.tool_calls = list(accumulated_tool_calls.values())

            case "message_delta":
                if event.delta.stop_reason:
                    stream_chunk.is_final = True

        if stream_chunk.content or stream_chunk.tool_calls or stream_chunk.is_final:
            yield stream_chunk
```

**注意事项：**
- Anthropic 的 `content_block_delta` 带 `index` 字段，关联到对应的 content block
- `input_json_delta.partial_json` 是 JSON 片段的增量，需拼接
- `content_block_start` 提供 tool_call 的 id/name，`input_json_delta` 提供 input，`content_block_stop` 结束
- 需要维护 `content_block_start` 到 `content_block_delta` 的关联（通过 index）

### 3.3 两种 Provider 流式行为对比

| 维度 | OpenAI | Anthropic |
|-----|--------|-----------|
| 流式接口 | `stream=True` + 迭代 chunk | `stream=True` + 迭代 event |
| 文本传输 | `delta.content`（增量） | `content_block_delta.text_delta`（增量） |
| 工具调用传输 | `delta.tool_calls`（完整快照+增量） | `content_block_start` + `input_json_delta`（分段） |
| 结束信号 | `finish_reason` | `message_delta.stop_reason` |
| 工具调用累积 | 按 index 拼接 arguments | 按 index 拼接 partial_json |

---

## 4. 文件变更清单

| 文件 | 变更类型 | 说明 |
|-----|---------|-----|
| `src/kocor/message.py` | 新增 | 新增 `StreamChunk` dataclass |
| `src/kocor/llm_client.py` | 修改 | `LLMClient` Protocol 新增 `stream()` 方法 |
| `src/kocor/openai_client.py` | 修改 | `OpenAIClient` 新增 `stream()` 实现 |
| `src/kocor/anthropic_client.py` | 修改 | `AnthropicClient` 新增 `stream()` 实现 |
| `src/kocor/agent.py` | 修改 | `Agent` 新增 `stream()` 方法 |
| `src/kocor/__main__.py` | 修改 | 新增 `--stream` CLI 参数 |
| `tests/test_message.py` | 新增 | `StreamChunk` 单元测试 |
| `tests/test_llm_client.py` | 修改 | Protocol 新增 `stream()` 测试 |
| `tests/test_openai_client.py` | 新增 | `stream()` 测试（文本+工具调用） |
| `tests/test_anthropic_client.py` | 新增 | `stream()` 测试（文本+工具调用） |
| `tests/test_agent.py` | 修改 | `Agent.stream()` 测试 |
| `docs/streaming_design.md` | 新增 | 本文档 |

---

## 5. TDD 实施计划

按 CLAUDE.md 要求，采用 Red-Green-Refactor 循环。

### Phase 1: 数据模型（message.py）

| 步骤 | 操作 |
|-----|------|
| Red | 写 `test_message.py::TestStreamChunk` — 测试默认值、属性访问、可实例化 |
| Green | 实现 `StreamChunk` dataclass |
| Refactor | 确认代码简洁，无冗余 |

### Phase 2: Protocol 扩展（llm_client.py）

| 步骤 | 操作 |
|-----|------|
| Red | 写 `test_llm_client.py::TestLLMClientProtocol` — 验证 `stream()` 签名、`create_llm_client` 返回的实例有 `stream` 方法 |
| Green | 在 `LLMClient` Protocol 中添加 `stream()` 方法 |
| Refactor | 确认无破坏性变更 |

### Phase 3: OpenAI 流式实现（openai_client.py）

| 步骤 | 操作 |
|-----|------|
| Red | 写 `test_openai_client.py::TestOpenAIClientStream` — 文本流式、工具调用流式、is_final 标记 |
| Green | 实现 `OpenAIClient.stream()` |
| Refactor | 提取累积逻辑为私有方法 |

### Phase 4: Anthropic 流式实现（anthropic_client.py）

| 步骤 | 操作 |
|-----|------|
| Red | 写 `test_anthropic_client.py::TestAnthropicClientStream` — 文本流式、工具调用流式、is_final 标记 |
| Green | 实现 `AnthropicClient.stream()` |
| Refactor | 提取事件处理逻辑为私有方法 |

### Phase 5: Agent 流式（agent.py）

| 步骤 | 操作 |
|-----|------|
| Red | 写 `test_agent.py::TestAgentStream` — 文本流式输出、工具调用后继续流式、is_final 标记 |
| Green | 实现 `Agent.stream()` |
| Refactor | 确认与 `run()` 共享消息管理逻辑 |

### Phase 6: CLI 改造（__main__.py）

| 步骤 | 操作 |
|-----|------|
| Red | 写集成测试 — `--stream` 参数解析、流式输出行为 |
| Green | 实现 `--stream` 参数和流式打印 |
| Refactor | 确认默认行为不变 |

---

## 6. 边界情况与处理

| 场景 | 处理方式 |
|-----|---------|
| LLM 返回空响应 | yield 一个 `is_final=True` 的 chunk |
| 工具调用参数 JSON 解析失败 | 在 `Agent.run()` 现有错误处理中捕获（不新增逻辑） |
| 流式过程中网络中断 | SDK 层抛出异常，向上冒泡，由调用方处理 |
| Agent 超时 | `Agent.stream()` 循环次数与 `run()` 一致，超时 yield 错误信息 chunk |
| 多工具调用 | OpenAI 的 `delta.tool_calls` 带 index；Anthropic 的 `content_block_start` 带 index，均可区分 |
| 工具执行结果过长 | 无特殊处理，作为普通文本追加到 messages，与现有行为一致 |

---

## 7. 不做什么（明确排除）

- **不改造现有 `generate()` 方法** — 保持同步 API 不变
- **不引入 async/await** — 流式通过 generator/iterator 实现，非异步
- **不支持 partial tool 结果流式** — 工具执行仍为阻塞，结果一次性返回
- **不改变 Agent 循环逻辑** — 流式仅改变输出方式，循环控制逻辑不变
- **不支持中断/取消流式** — 无 CancellationToken，简单优先
- **不添加重试/断线重连** — 超出本次范围

---

## 8. 风险评估

| 风险 | 影响 | 缓解 |
|-----|------|-----|
| Anthropic 流式事件处理复杂 | 实现 bug 概率较高 | 充分单元测试覆盖各事件类型 |
| `StreamChunk` 设计不够灵活 | 后续扩展受限 | 当前设计已预留 tool_calls 字段，可扩展 |
| CLI 流式输出与工具调用输出混排 | 终端显示混乱 | 工具调用用独立行+前缀，清晰区分 |
| 测试 mock 流式 API | 测试复杂度增加 | 为每个 provider 设计专用 mock 类 |

---

## 9. 预期效果

**改造前：**
```
$ python -m kocor "帮我读 test.txt 并统计字数"
[等待 3 秒]
文件内容是: hello world
共 2 个单词
```

**改造后（--stream）：**
```
$ python -m kocor --stream "帮我读 test.txt 并统计字数"
文件内容是: hello world[工具调用] read_file({"path": "test.txt"})
[工具执行中...]
共 2 个单词
```

文字在 LLM 思考时逐字出现，工具调用有明确标记，工具执行后继续输出。感知延迟显著降低。
