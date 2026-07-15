"""测试公共辅助。"""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def chdir_cm(path: str):
    """临时切换工作目录，确保退出时恢复。

    文件工具以 os.getcwd() 为允许目录，测试需 chdir 到临时目录才能让
    绝对路径通过越界检查。必须在 TemporaryDirectory 退出前恢复 cwd，
    否则 Windows 无法删除被进程占用的当前目录。
    """
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
