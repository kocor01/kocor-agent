"""Harness 运行时的调试支持。"""

import json


class DebugManager:
    """在调试模式启用时提供详细的运行时信息。

    记录事件、打印上下文快照、显示工具调用历史以诊断 Agent 行为。
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events: list = []

    def record_event(self, event) -> None:
        """记录一个 harness 事件（仅在启用时）。"""
        if not self.enabled:
            return
        self.events.append(event)

    def clear(self) -> None:
        """清除所有已记录的事件。"""
        self.events.clear()

    def print_context(self, messages: list) -> None:
        """打印当前消息上下文的摘要。"""
        if not self.enabled:
            return
        total_tokens = sum(len(str(m.content)) // 4 for m in messages if m.content)
        print(f"\n{'─' * 50}")
        print(f"[DEBUG] 上下文概览")
        print(f"  消息数: {len(messages)}")
        print(f"  估算 Token: ~{total_tokens}")
        if messages and messages[0].role == "system":
            print(f"  System Prompt: {len(str(messages[0].content))} chars")
        for msg in messages[-3:]:
            preview = (str(msg.content)[:100] + "...") if msg.content and len(str(msg.content)) > 100 else (str(msg.content) or "")
            tool_info = ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_info = f" [tools: {', '.join(tc.function.name for tc in msg.tool_calls)}]"
            print(f"  [{msg.role}]{tool_info}: {preview}")
        print(f"{'─' * 50}\n")

    def print_tool_history(self, records: list) -> None:
        """打印工具调用历史摘要。"""
        if not self.enabled or not records:
            return
        print(f"\n{'─' * 50}")
        print(f"[DEBUG] 工具调用历史 ({len(records)} 次)")
        for rec in records:
            icon = "OK" if rec.error is None else "ERR"
            perm_icon = "AUTO" if rec.permission == "auto" else "CONFIRM" if rec.permission == "confirm" else "DENY"
            args_str = json.dumps(rec.arguments)[:80] if hasattr(rec, "arguments") and rec.arguments else "{}"
            print(f"  #{rec.iteration} {icon} {perm_icon} {rec.tool_name}({args_str})")
            print(f"     耗时: {rec.duration_ms:.0f}ms | 结果: {str(rec.result_summary)[:100]}")
        print(f"{'─' * 50}\n")