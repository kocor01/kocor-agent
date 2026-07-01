"""覆盖所有工具类型的统一权限系统。

用三层策略模型（宽松/默认/严格）取代仅限 MCP 的 PermissionManager，
适用于内置工具、MCP 工具和技能工具。
"""

import json


class PermissionManager:
    """所有工具类型的统一权限管理器。

    三层策略：
    - permissive: 全部自动允许（不确认）
    - default: safe 自动允许，caution/dangerous 询问一次
    - strict: 全部检查，dangerous 默认拒绝

    安全等级（每个工具注册时指定）：
    - safe:      只读操作，无害（如读文件），所有策略下均自动放行
    - caution:   有潜在影响的操作（如目录列举），default/strict 会询问
    - dangerous: 高风险操作（如写文件、执行代码），strict 下默认拒绝

    会话级别的缓存避免同一工具的重复提示。
    """

    # 权限策略常量
    POLICY_PERMISSIVE = "permissive"
    POLICY_DEFAULT = "default"
    POLICY_STRICT = "strict"

    # 安全等级常量
    SAFETY_SAFE = "safe"
    SAFETY_CAUTION = "caution"
    SAFETY_DANGEROUS = "dangerous"

    def __init__(
        self,
        policy: str = POLICY_DEFAULT,
        always_allow: set[str] | None = None,
        always_ask: set[str] | None = None,
        cache_enabled: bool = True,
        cache_max_size: int = 50,
        tool_manager=None,
    ):
        self.policy = policy
        self._always_allow = always_allow or set()
        self._always_ask = always_ask or set()
        self._cache: set[str] = set()
        self.cache_enabled = cache_enabled
        self.cache_max_size = cache_max_size
        self._tool_manager = tool_manager

    def check(self, tool_call) -> bool:
        """检查工具调用是否应被允许。

        返回 True 表示允许，False 表示拒绝。当需要用户确认时，
        通过 stdin 提示。
        """
        tool_name = tool_call.function.name
        args = tool_call.function.arguments

        # 1. 始终允许列表覆盖一切
        if tool_name in self._always_allow:
            return True

        # 2. 会话缓存（之前已批准）
        if self.cache_enabled and tool_name in self._cache:
            return True

        # 3. 始终询问列表强制提示
        if tool_name in self._always_ask:
            return self._ask_user(tool_name, args)

        # 4. 基于策略的决策
        safety = getattr(self._tool_manager._tools.get(tool_name), 'safety_level', PermissionManager.SAFETY_CAUTION) if self._tool_manager else PermissionManager.SAFETY_CAUTION

        if self.policy == PermissionManager.POLICY_PERMISSIVE:
            return True

        if self.policy == PermissionManager.POLICY_STRICT:
            if safety == PermissionManager.SAFETY_SAFE:
                return True
            if safety == PermissionManager.SAFETY_DANGEROUS:
                return False
            return self._ask_user(tool_name, args)

        # default 策略
        if safety == PermissionManager.SAFETY_SAFE:
            return True
        return self._ask_user(tool_name, args)

    def _ask_user(self, tool_name: str, args: dict | None = None) -> bool:
        """提示用户确认工具调用。

        返回 True 表示已批准（包括"始终允许此会话"）。
        如果 stdin 不可用（非交互模式），返回 False。
        """
        try:
            print(f"⚠️  工具调用需要确认: ")
            print(f"   名称: {tool_name}")
            if args:
                print(f"   参数: {args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)[:200]}")
            response = input("   允许执行? (Y/n/a=始终允许此会话): ").strip().lower()
        except (EOFError, OSError, KeyboardInterrupt):
            return False

        if response in ("a", "always"):
            self._add_to_cache(tool_name)
            return True
        if response in ("", "y", "yes"):
            self._add_to_cache(tool_name)
            return True
        return False

    def _add_to_cache(self, tool_name: str) -> None:
        """将工具添加到会话缓存，遵守最大大小限制。"""
        if len(self._cache) >= self.cache_max_size:
            self._cache.pop()
        self._cache.add(tool_name)

    def clear_cache(self) -> None:
        """清除会话级别的批准缓存。"""
        self._cache.clear()

    def update_config(self, config: dict) -> None:
        """从字典更新配置（例如解析后的 JSON）。"""
        if "policy" in config:
            policy = config["policy"]
            self.policy = policy
        if "always_allow" in config:
            self._always_allow = set(config["always_allow"])
        if "always_ask" in config:
            self._always_ask = set(config["always_ask"])