"""todo 任务规划工具。

轻量、内存态的任务列表，用于 LLM 分解复杂任务、追踪进度、跨上下文压缩恢复焦点。
状态附着在 Agent 实例上（一个会话一个），随会话生灭。

设计（沿用 hermes 已验证模型）：
- 单一 todo 工具：传 todos 写入，不传读取
- 4 态状态机：pending → in_progress → completed/cancelled
- 列表顺序即优先级（无显式 priority 字段）
- 双写入模式：merge=false 整表替换，merge=true 按 id 更新/追加
- 防御性编程：LLM 输出不可预测，每一层优雅降级、永不崩溃
- 上下文压缩后只注入 active 项，避免 LLM 重做已完成任务
"""

from __future__ import annotations

import json
from typing import Any

from kocor.llm_provider.message import Message
from kocor.tools.permission import PermissionManager

# 合法状态值
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# 安全边界（非用户可调行为，作为模块常量，与 MemoryStore 内部常量风格一致）。
# todo 列表是规划辅助，会在上下文压缩后重新注入（见 format_for_injection），
# 无界的内容或数量会抵消压缩效果。这些上限相对真实计划足够宽裕。
MAX_TODO_CONTENT_CHARS = 4000      # 单条 content 字符上限
MAX_TODO_ITEMS = 256               # 列表条数上限
# hydrate 时单条 tool 结果解析上限，防止伪造超大 payload 被解析后再注入。
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"

# 注入块的状态标记
_MARKERS = {
    "completed": "[x]",
    "in_progress": "[>]",
    "pending": "[ ]",
    "cancelled": "[~]",
}


class TodoStore:
    """内存态任务列表，一个 Agent 实例持有一个。

    条目为扁平 dict：{id, content, status}。列表顺序即优先级。
    """

    def __init__(self):
        self._items: list[dict[str, str]] = []

    def write(self, todos: list[dict[str, Any]], merge: bool = False) -> list[dict[str, str]]:
        """写入任务列表，返回写入后的完整列表（read() 副本）。

        Args:
            todos: {id, content, status} dict 列表
            merge: False（默认）整表替换；True 按 id 更新已有项、追加新项
        """
        if not merge:
            # 替换模式：去重后逐条校验，整表替换
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
        else:
            # 合并模式：按 id 更新已存在项的 content/status，追加新项
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip() if isinstance(t, dict) else ""
                if not item_id:
                    continue  # 合并必须有 id
                if item_id in existing:
                    # 只更新明确提供的字段，保留未提供的字段
                    if isinstance(t.get("content"), str) and t["content"].strip():
                        existing[item_id]["content"] = self._cap_content(t["content"].strip())
                    if t.get("status"):
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    # 新项：完整校验后追加
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # 按原顺序重建，去重
            seen: set[str] = set()
            rebuilt: list[dict[str, str]] = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt

        # 截断到数量上限，保留头部（高优先级）
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        return self.read()

    def read(self) -> list[dict[str, str]]:
        """返回列表浅拷贝，防止外部修改内部状态。"""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """列表是否非空。"""
        return bool(self._items)

    def format_for_injection(self) -> str | None:
        """渲染 active 项供上下文压缩后注入。

        只保留 pending/in_progress；已完成/已取消不注入（避免 LLM 重做）。
        无 active 项返回 None。
        """
        if not self._items:
            return None
        active = [it for it in self._items if it["status"] in {"pending", "in_progress"}]
        if not active:
            return None
        lines = ["[Your active task list was preserved across context compression]"]
        for it in active:
            marker = _MARKERS.get(it["status"], "[?]")
            lines.append(f"- {marker} {it['id']}. {it['content']} ({it['status']})")
        return "\n".join(lines)

    def hydrate_from_history(self, messages: list[Message]) -> None:
        """从会话历史回填 store（取最后一次有效 todo 结果）。

        kocor 的 tool 消息只带 tool_call_id，工具名需通过前一条 assistant 的
        tool_calls 关联。超大结果跳过。回填数据复用 write 的校验链。
        """
        # 建立 tool_call_id -> tool_name 映射（仅 todo 工具）
        todo_call_ids: set[str] = set()
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == "todo":
                        todo_call_ids.add(tc.id)

        # 找到最后一个有效的 todo tool 结果
        last_items: list[dict[str, Any]] | None = None
        for msg in messages:
            if msg.role != "tool" or not msg.tool_call_id:
                continue
            if msg.tool_call_id not in todo_call_ids:
                continue
            if not msg.content or len(msg.content) > MAX_TODO_RESULT_CHARS:
                continue
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict) and isinstance(data.get("todos"), list):
                last_items = data["todos"]

        if last_items:
            self.write(last_items, merge=False)

    @staticmethod
    def _cap_content(content: str) -> str:
        """截断超长 content，保留头部（可执行部分）+ 标记。"""
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Any) -> dict[str, str]:
        """校验并规整单条任务，永不崩溃。

        非 dict → 占位项；空 id → "?"；空 content → 占位；非法 status → pending。
        返回仅含 {id, content, status} 的干净 dict。
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"
        else:
            content = TodoStore._cap_content(content)

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe_by_id(todos: list[Any]) -> list[Any]:
        """同 id 保留最后一次出现位置；非 dict 用合成键。"""
        last_index: dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]


class TodoTool:
    """任务规划工具（仿 MemoryTool 类式风格）。"""

    @classmethod
    def handler_factory(cls, **deps):
        """返回带 store 注入的 handler（延迟解析，因为 store 注册后才设定）。"""
        tm = deps.get("tool_manager")
        return lambda **kw: TodoTool.handler(store=tm.todo_store if tm else None, **kw)

    NAME = "todo"
    DESCRIPTION = """管理当前会话的任务列表。WHEN: 复杂任务（3+ 步）或用户提供多个任务时主动规划。
HOW: 不传 todos 读取当前列表；传 todos 写入。merge=false(默认)整体替换；merge=true 按 id 更新已有项、追加新项。
每项 {id, content, status: pending|in_progress|completed|cancelled}。列表顺序即优先级，同时只允许一个 in_progress。
完成立即标记 completed；失败则 cancelled 并新增修正项。每次调用返回完整列表。"""
    SAFETY_LEVEL = PermissionManager.SAFETY_SAFE  # 纯内存，无副作用

    PARAMETERS = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "要写入的任务项。省略则读取当前列表。",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "任务唯一标识"},
                        "content": {"type": "string", "description": "任务描述"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "当前状态",
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            },
            "merge": {
                "type": "boolean",
                "description": "true: 按 id 更新已有项、追加新项。false(默认): 整体替换。",
                "default": False,
            },
        },
        "required": [],
    }

    @staticmethod
    def handler(store: TodoStore | None, todos: Any = None, merge: bool = False) -> str:
        """处理 todo 工具调用。

        Args:
            store: TodoStore 实例
            todos: 任务项列表（可能是数组或 JSON 字符串）；None 表示读取
            merge: 是否合并模式

        Returns:
            JSON 字符串 {todos, summary} 或 {success, error}
        """
        if store is None:
            return json.dumps(
                {"success": False, "error": "todo store not available"}, ensure_ascii=False
            )

        if todos is not None:
            # LLM 有时把 todos 序列化为 JSON 字符串而非数组，兼容解析
            if isinstance(todos, str):
                try:
                    todos = json.loads(todos)
                except (json.JSONDecodeError, TypeError):
                    return json.dumps(
                        {"success": False, "error": "todos must be a list of objects, got unparseable string"},
                        ensure_ascii=False,
                    )
            if not isinstance(todos, list):
                return json.dumps(
                    {"success": False, "error": f"todos must be a list, got {type(todos).__name__}"},
                    ensure_ascii=False,
                )
            items = store.write(todos, merge)
        else:
            items = store.read()

        summary = {
            "total": len(items),
            "pending": sum(1 for i in items if i["status"] == "pending"),
            "in_progress": sum(1 for i in items if i["status"] == "in_progress"),
            "completed": sum(1 for i in items if i["status"] == "completed"),
            "cancelled": sum(1 for i in items if i["status"] == "cancelled"),
        }
        return json.dumps({"todos": items, "summary": summary}, ensure_ascii=False)
