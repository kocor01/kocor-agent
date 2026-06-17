"""后台事件循环，桥接异步 MCP SDK 到同步接口。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError
from typing import Any

_mcp_loop: asyncio.AbstractEventLoop | None = None
_mcp_loop_lock = threading.Lock()


class MCPError(Exception):
    """MCP 协议错误。"""
    pass


def _get_loop() -> asyncio.AbstractEventLoop:
    """获取或创建后台 asyncio 事件循环（守护线程）。"""
    global _mcp_loop
    if _mcp_loop is None:
        with _mcp_loop_lock:
            if _mcp_loop is None:
                _mcp_loop = asyncio.new_event_loop()
                t = threading.Thread(
                    target=_mcp_loop.run_forever,
                    daemon=True,
                    name="mcp-event-loop",
                )
                t.start()
    return _mcp_loop


def _run_async(coro, timeout: float = 120) -> Any:
    """在后台事件循环上运行协程，同步等待结果。

    Args:
        coro: 要执行的协程
        timeout: 超时秒数

    Returns:
        协程返回值

    Raises:
        MCPError: 超时或执行出错
    """
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        raise MCPError(f"Async operation timed out after {timeout}s")
    except Exception as e:
        if isinstance(e, MCPError):
            raise
        raise MCPError(str(e)) from e
