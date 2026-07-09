"""测试 todo 任务规划工具。"""

from __future__ import annotations

import json

import pytest

from kocor.llm_provider.message import FunctionCall, Message, ToolCall
from kocor.tools.permission import PermissionManager
from kocor.tools.toolsets.todo_tool import (
    MAX_TODO_CONTENT_CHARS,
    MAX_TODO_ITEMS,
    TodoStore,
    TodoTool,
)


# ──────────────────────────── TodoStore: 读写 ────────────────────────────


class TestTodoStoreWriteRead:
    """测试 TodoStore 读写。"""

    def test_write_replace_returns_full_list(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "Write report", "status": "pending"},
            {"id": "2", "content": "Review PR", "status": "in_progress"},
        ]
        result = store.write(items)
        assert len(result) == 2
        assert result[0]["id"] == "1"

    def test_read_returns_shallow_copy(self):
        """修改 read() 返回的副本不应影响内部状态。"""
        store = TodoStore()
        store.write([{"id": "1", "content": "task", "status": "pending"}])
        snapshot = store.read()
        snapshot[0]["status"] = "completed"
        # 内部状态未被改动
        assert store.read()[0]["status"] == "pending"

    def test_read_empty_store(self):
        assert TodoStore().read() == []

    def test_has_items(self):
        store = TodoStore()
        assert store.has_items() is False
        store.write([{"id": "1", "content": "t", "status": "pending"}])
        assert store.has_items() is True

    def test_write_replace_overwrites_previous(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "old", "status": "pending"}])
        store.write([{"id": "2", "content": "new", "status": "pending"}])
        result = store.read()
        assert len(result) == 1
        assert result[0]["id"] == "2"


# ──────────────────────────── TodoStore: 合并模式 ────────────────────────────


class TestTodoStoreMerge:
    """测试合并写入模式。"""

    def test_merge_updates_existing_status(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task A", "status": "pending"},
            {"id": "2", "content": "task B", "status": "pending"},
        ])
        # 只更新 id=1 的 status，不提供 content
        store.write([{"id": "1", "status": "completed"}], merge=True)
        result = store.read()
        assert result[0]["status"] == "completed"
        # content 被保留
        assert result[0]["content"] == "task A"
        # 其他项不动
        assert result[1]["content"] == "task B"

    def test_merge_appends_new_item(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task A", "status": "pending"}])
        store.write([{"id": "2", "content": "task B", "status": "pending"}], merge=True)
        result = store.read()
        assert len(result) == 2
        assert result[1]["id"] == "2"

    def test_merge_updates_content_partial(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "old", "status": "in_progress"}])
        store.write([{"id": "1", "content": "new content"}], merge=True)
        result = store.read()
        assert result[0]["content"] == "new content"
        # status 未提供，保留原值
        assert result[0]["status"] == "in_progress"

    def test_merge_without_id_is_ignored(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "task A", "status": "pending"}])
        store.write([{"content": "no id", "status": "pending"}], merge=True)
        assert len(store.read()) == 1


# ──────────────────────────── TodoStore: 防御性校验 ────────────────────────────


class TestTodoStoreValidation:
    """测试 _validate 的优雅降级。"""

    def test_non_dict_becomes_placeholder(self):
        result = TodoStore._validate("not a dict")
        assert result["id"] == "?"
        assert result["status"] == "pending"

    def test_none_becomes_placeholder(self):
        result = TodoStore._validate(None)
        assert result["id"] == "?"
        assert result["status"] == "pending"

    def test_empty_id_becomes_question(self):
        result = TodoStore._validate({"content": "t", "status": "pending"})
        assert result["id"] == "?"

    def test_empty_content_becomes_placeholder(self):
        result = TodoStore._validate({"id": "1", "status": "pending"})
        assert result["content"] == "(no description)"

    def test_invalid_status_falls_back_to_pending(self):
        result = TodoStore._validate({"id": "1", "content": "t", "status": "bogus"})
        assert result["status"] == "pending"

    def test_valid_item_preserved(self):
        result = TodoStore._validate({"id": "1", "content": "t", "status": "completed"})
        assert result == {"id": "1", "content": "t", "status": "completed"}


class TestTodoStoreDedupe:
    """测试 _dedupe_by_id 保留最后出现。"""

    def test_dedupe_keeps_last_occurrence(self):
        items = [
            {"id": "1", "content": "first", "status": "pending"},
            {"id": "1", "content": "second", "status": "in_progress"},
        ]
        deduped = TodoStore._dedupe_by_id(items)
        assert len(deduped) == 1
        assert deduped[0]["content"] == "second"

    def test_dedupe_preserves_position_of_last(self):
        items = [
            {"id": "1", "content": "a", "status": "pending"},
            {"id": "2", "content": "b", "status": "pending"},
            {"id": "1", "content": "a2", "status": "pending"},
        ]
        deduped = TodoStore._dedupe_by_id(items)
        # id=1 保留在最后出现的位置（index 2），id=2 保留在 index 1
        assert [d["id"] for d in deduped] == ["2", "1"]

    def test_dedupe_handles_non_dict(self):
        items = ["str", {"id": "1", "content": "t", "status": "pending"}]
        deduped = TodoStore._dedupe_by_id(items)
        assert len(deduped) == 2


# ──────────────────────────── TodoStore: 边界 ────────────────────────────


class TestTodoStoreBounds:
    """测试内容截断与数量上限。"""

    def test_cap_content_under_limit(self):
        assert TodoStore._cap_content("short") == "short"

    def test_cap_content_over_limit_truncates(self):
        long_content = "x" * (MAX_TODO_CONTENT_CHARS + 1000)
        capped = TodoStore._cap_content(long_content)
        assert len(capped) == MAX_TODO_CONTENT_CHARS
        assert capped.endswith("… [truncated]")

    def test_write_caps_oversized_content(self):
        long_content = "y" * (MAX_TODO_CONTENT_CHARS + 500)
        store = TodoStore()
        store.write([{"id": "1", "content": long_content, "status": "pending"}])
        item = store.read()[0]
        assert len(item["content"]) == MAX_TODO_CONTENT_CHARS

    def test_write_truncates_to_max_items(self):
        items = [
            {"id": str(i), "content": f"task {i}", "status": "pending"}
            for i in range(MAX_TODO_ITEMS + 50)
        ]
        store = TodoStore()
        store.write(items)
        assert len(store.read()) == MAX_TODO_ITEMS


# ──────────────────────────── TodoStore: 注入渲染 ────────────────────────────


class TestTodoStoreInjection:
    """测试 format_for_injection。"""

    def test_empty_store_returns_none(self):
        assert TodoStore().format_for_injection() is None

    def test_only_completed_returns_none(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "done", "status": "completed"},
            {"id": "2", "content": "cancelled one", "status": "cancelled"},
        ])
        assert store.format_for_injection() is None

    def test_filters_completed_and_cancelled(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "active", "status": "in_progress"},
            {"id": "2", "content": "done", "status": "completed"},
            {"id": "3", "content": "cancelled one", "status": "cancelled"},
            {"id": "4", "content": "pending one", "status": "pending"},
        ])
        text = store.format_for_injection()
        assert text is not None
        assert "active" in text
        assert "pending one" in text
        assert "done" not in text
        assert "cancelled one" not in text

    def test_injection_format_and_markers(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Write report", "status": "in_progress"},
            {"id": "2", "content": "Review PR", "status": "pending"},
        ])
        text = store.format_for_injection()
        lines = text.split("\n")
        assert "compression" in lines[0].lower()
        assert "- [>] 1. Write report (in_progress)" in lines
        assert "- [ ] 2. Review PR (pending)" in lines


# ──────────────────────────── TodoStore: hydrate ────────────────────────────


def _todo_result_message(todos: list[dict]) -> str:
    """构造 todo 工具的 tool 结果 JSON。"""
    return json.dumps({"todos": todos, "summary": {}}, ensure_ascii=False)


class TestTodoStoreHydrate:
    """测试从历史消息回填。"""

    def test_hydrate_from_history_last_result(self):
        """应回填最后一次有效 todo 结果。"""
        store = TodoStore()
        messages = [
            Message(role="user", content="start"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(name="todo", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c1", content=_todo_result_message([
                {"id": "1", "content": "first plan", "status": "pending"},
            ])),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c2", function=FunctionCall(name="todo", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c2", content=_todo_result_message([
                {"id": "2", "content": "second plan", "status": "in_progress"},
                {"id": "3", "content": "third", "status": "pending"},
            ])),
        ]
        store.hydrate_from_history(messages)
        result = store.read()
        assert len(result) == 2
        assert result[0]["id"] == "2"
        assert result[0]["status"] == "in_progress"

    def test_hydrate_no_todo_calls_keeps_empty(self):
        store = TodoStore()
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        store.hydrate_from_history(messages)
        assert store.has_items() is False

    def test_hydrate_skips_oversized_result(self):
        """超大 tool 结果应被跳过，回退到前一个有效结果。"""
        store = TodoStore()
        big_payload = "x" * 600_000
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(name="todo", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c1", content=big_payload),
        ]
        store.hydrate_from_history(messages)
        assert store.has_items() is False

    def test_hydrate_ignores_non_todo_tool_results(self):
        """非 todo 工具的结果不应被解析。"""
        store = TodoStore()
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c1", content=_todo_result_message([
                {"id": "1", "content": "should be ignored", "status": "pending"},
            ])),
        ]
        store.hydrate_from_history(messages)
        assert store.has_items() is False

    def test_hydrate_reuses_validation_chain(self):
        """回填的数据应经过 _validate 校验（非法 status 回退 pending）。"""
        store = TodoStore()
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", function=FunctionCall(name="todo", arguments='{}'))],
            ),
            Message(role="tool", tool_call_id="c1", content=_todo_result_message([
                {"id": "1", "content": "t", "status": "bogus"},
            ])),
        ]
        store.hydrate_from_history(messages)
        assert store.read()[0]["status"] == "pending"


# ──────────────────────────── TodoTool: 工具定义 ────────────────────────────


class TestTodoToolDefinition:
    """测试工具定义常量。"""

    def test_name_and_description(self):
        assert TodoTool.NAME == "todo"
        assert TodoTool.DESCRIPTION

    def test_parameters_schema(self):
        props = TodoTool.PARAMETERS["properties"]
        assert "todos" in props
        assert "merge" in props
        assert TodoTool.PARAMETERS["required"] == []
        # todos.items 必填字段
        item_props = props["todos"]["items"]["properties"]
        assert set(item_props.keys()) == {"id", "content", "status"}
        assert set(item_props["status"]["enum"]) == {"pending", "in_progress", "completed", "cancelled"}

    def test_safety_level_is_safe(self):
        assert TodoTool.SAFETY_LEVEL == PermissionManager.SAFETY_SAFE


# ──────────────────────────── TodoTool: handler ────────────────────────────


class TestTodoToolHandler:
    """测试 handler 的输入输出。"""

    def test_handler_store_none_returns_error(self):
        result = TodoTool.handler(store=None, todos=[{"id": "1", "content": "t", "status": "pending"}])
        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data

    def test_handler_write_returns_json_with_summary(self):
        store = TodoStore()
        result = TodoTool.handler(
            store=store,
            todos=[
                {"id": "1", "content": "a", "status": "pending"},
                {"id": "2", "content": "b", "status": "in_progress"},
                {"id": "3", "content": "c", "status": "completed"},
                {"id": "4", "content": "d", "status": "cancelled"},
            ],
        )
        data = json.loads(result)
        assert len(data["todos"]) == 4
        assert data["summary"] == {
            "total": 4, "pending": 1, "in_progress": 1, "completed": 1, "cancelled": 1,
        }

    def test_handler_read_mode(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "t", "status": "pending"}])
        result = TodoTool.handler(store=store)
        data = json.loads(result)
        assert len(data["todos"]) == 1
        assert data["summary"]["total"] == 1

    def test_handler_read_empty_store(self):
        store = TodoStore()
        result = TodoTool.handler(store=store)
        data = json.loads(result)
        assert data["todos"] == []
        assert data["summary"]["total"] == 0

    def test_handler_todos_as_json_string(self):
        """LLM 把 todos 序列化成 JSON 字符串时应兼容。"""
        store = TodoStore()
        todos_str = json.dumps([{"id": "1", "content": "from string", "status": "pending"}])
        result = TodoTool.handler(store=store, todos=todos_str)
        data = json.loads(result)
        assert data["summary"]["total"] == 1
        assert data["todos"][0]["content"] == "from string"

    def test_handler_todos_unparseable_string(self):
        store = TodoStore()
        result = TodoTool.handler(store=store, todos="not json")
        data = json.loads(result)
        assert data["success"] is False

    def test_handler_todos_not_list(self):
        store = TodoStore()
        result = TodoTool.handler(store=store, todos=42)
        data = json.loads(result)
        assert data["success"] is False

    def test_handler_merge_mode(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "t", "status": "pending"}])
        result = TodoTool.handler(
            store=store, todos=[{"id": "1", "status": "completed"}], merge=True,
        )
        data = json.loads(result)
        assert data["todos"][0]["status"] == "completed"
        assert data["summary"]["completed"] == 1

    def test_handler_unicode_preserved(self):
        store = TodoStore()
        result = TodoTool.handler(
            store=store, todos=[{"id": "1", "content": "写报告", "status": "pending"}],
        )
        # ensure_ascii=False 应保留中文原文
        assert "写报告" in result
