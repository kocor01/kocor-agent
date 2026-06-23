"""MCP 服务器配置加载。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPConfig:
    """单个 MCP 服务器的配置。

    stdio 服务器使用 command/args/env。
    远程服务器使用 url/headers/transport。
    """

    # stdio 字段
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # 远程字段
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    transport: str = "streamablehttp"  # "streamablehttp" | "sse"
    # 超时
    timeout: int = 120
    connect_timeout: int = 30


def load_mcp_servers(config_path: str) -> dict[str, MCPConfig]:
    """从 JSON 配置文件加载 MCP 服务器列表。"""
    if not config_path or not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    servers: dict[str, Any] = data.get("mcpServers", {})
    result: dict[str, MCPConfig] = {}
    for name, cfg in servers.items():
        result[name] = MCPConfig(
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            env=cfg.get("env", {}),
            url=cfg.get("url", ""),
            headers=cfg.get("headers", {}),
            transport=cfg.get("transport", "streamablehttp"),
            timeout=cfg.get("timeout", 120),
            connect_timeout=cfg.get("connect_timeout", 30),
        )
    return result


def sanitize_server_name(name: str) -> str:
    """将服务器/工具名称清洗为安全的 Python 标识符。"""
    result: list[str] = []
    for ch in name.lower():
        if ch.isalnum() or ch == "_":
            result.append(ch)
        else:
            result.append("_")
    return "".join(result)
