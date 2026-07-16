"""memory 内置工具。

LLM 通过此工具管理长期记忆（MEMORY.md / USER.md）。
采用批量 operations 数组，一次性完成多个操作（全有或全无）。
"""

from __future__ import annotations

import json
from typing import Any

from kocor.memory.store import MemoryOp, MemoryStore
from kocor.memory.types import MemoryTarget
from kocor.tools.permission import PermissionManager


class MemoryTool:
    @classmethod
    def handler_factory(cls, **deps):
        """返回带 store 注入的 handler（延迟解析，因为 store 注册后才设定）。"""
        tm = deps.get("tool_manager")
        return lambda **kw: MemoryTool.handler(store=tm.memory_store if tm else None, **kw)


    """管理长期记忆。"""

    NAME = "memory"
    DESCRIPTION = """管理长期记忆。WHEN: 用户表达偏好/纠正/个人细节，或揭示稳定的环境事实时主动保存。
SKIP: 任务进度、PR 编号、commit SHA、7 天内会过时的内容。
HOW: 通过 operations 数组一次性批量操作（add/replace/remove），按最终状态校验容量。
IF FULL: 先 remove 腾空间，再 add。

IMPORTANT: 成功保存后不要再调用 memory。立即回复用户。"""
    SAFETY_LEVEL = PermissionManager.SAFETY_SAFE
    PARAMETERS = {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "replace", "remove"],
                            "description": "add: 追加新条目 | replace: 替换已有条目 | remove: 删除条目",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["memory", "user"],
                            "description": "memory: Agent 个人笔记 | user: 用户画像",
                        },
                        "content": {
                            "type": "string",
                            "description": "条目内容（add/replace 必填）",
                        },
                        "old_substring": {
                            "type": "string",
                            "description": "要替换/删除的条目的唯一子串（replace/remove 必填）",
                        },
                    },
                    "required": ["action", "target"],
                },
            },
        },
        "required": ["operations"],
    }

    @staticmethod
    def handler(store: MemoryStore, operations: Any) -> str:
        """处理 memory 工具调用。

        Args:
            store: MemoryStore 实例
            operations: 操作列表（可能是数组或 JSON 字符串）

        Returns:
            JSON 字符串
        """
        if not store:
            return json.dumps({"success": False, "error": "memory store not available"}, ensure_ascii=False)

        # LLM 有时将 operations 序列化为 JSON 字符串而非数组，做兼容解析
        if isinstance(operations, str):
            try:
                operations = json.loads(operations)
            except json.JSONDecodeError:
                return json.dumps({"success": False, "error": "operations is not valid JSON"}, ensure_ascii=False)

        if not operations:
            return json.dumps({"success": True}, ensure_ascii=False)

        try:
            ops = [_parse_op(op) for op in operations]
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

        # 单操作走单通道，批量走 apply_batch
        if len(ops) == 1:
            op = ops[0]
            result = store._apply_single(op)
        else:
            result = store.apply_batch(ops)

        response: dict[str, Any] = {"success": result.success}
        if result.error:
            response["error"] = result.error
        if result.target:
            response["target"] = result.target.value
        if result.current_entries:
            response["current_entries"] = result.current_entries
        if result.usage:
            response["usage"] = result.usage

        return json.dumps(response, ensure_ascii=False)


def _parse_op(d: dict) -> MemoryOp:
    """解析字典为 MemoryOp。"""
    action = d.get("action", "")
    if action not in ("add", "replace", "remove"):
        raise ValueError(f"unknown action: {action}")

    target_raw = d.get("target", "memory")
    try:
        target = MemoryTarget(target_raw)
    except ValueError:
        raise ValueError(f"unknown target: {target_raw}")

    return MemoryOp(
        action=action,
        target=target,
        content=d.get("content", ""),
        old_substring=d.get("old_substring", ""),
    )