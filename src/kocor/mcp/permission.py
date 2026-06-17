"""MCP 工具执行权限管理。

支持两种策略：
- always_allow: 自动放行（默认）
- always_ask: 每次执行前通过 stdin 询问用户确认
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PermissionPolicy:
    """单个 MCP 服务器的权限策略。"""
    policy: str = "always_allow"  # "always_allow" | "always_ask"
    allowed_tools: list[str] = field(default_factory=list)


class PermissionManager:
    """工具执行权限管理。

    流程:
    1. 检查工具是否在 allowed_tools 列表中 → 自动放行
    2. 检查 policy == always_allow → 自动放行
    3. policy == always_ask → 打印提示 → 等待 stdin 确认

    首次确认的工具在本会话内缓存，不再重复询问。
    """

    def __init__(self, policies: dict[str, PermissionPolicy]):
        self._policies = policies
        self._session_cache: set[str] = set()

    def check(self, tool_full_name: str, server_name: str) -> bool:
        """检查工具调用是否需要用户确认。

        Args:
            tool_full_name: 完整工具名（如 mcp_github_list_issues）
            server_name: MCP 服务器名

        Returns:
            True=允许执行，False=拒绝
        """
        policy = self._policies.get(server_name, PermissionPolicy())

        # 会话缓存
        if tool_full_name in self._session_cache:
            return True

        # allowed_tools 白名单
        if tool_full_name in policy.allowed_tools:
            self._session_cache.add(tool_full_name)
            return True

        # always_allow
        if policy.policy == "always_allow":
            return True

        # always_ask：询问用户
        print(f"\n⚠️  MCP 工具调用需要确认")
        print(f"   服务器: {server_name}")
        print(f"   工具:   {tool_full_name}")
        response = input("   允许执行? (Y/n/a=始终允许此会话): ").strip().lower()

        if response in ("a", "always"):
            self._session_cache.add(tool_full_name)
            return True
        if response in ("", "y", "yes"):
            self._session_cache.add(tool_full_name)
            return True
        return False
