"""文件访问安全守卫。

将文件读/写操作限制在允许的目录内，
并阻止对敏感文件（如 .env）的访问。
"""

import os


class FileAccessGuard:
    """限制工具文件 I/O 到允许的目录，并拦截敏感文件。"""

    def __init__(self, allowed_dir: str = ""):
        self.allowed_dir = os.path.abspath(allowed_dir) if allowed_dir else ""

    def check_read(self, path: str) -> str:
        """验证读取权限。返回标准化路径或抛出 PermissionError。"""
        abs_path = os.path.abspath(path)
        if self.allowed_dir and not abs_path.startswith(self.allowed_dir):
            raise PermissionError(
                f"读取 '{path}' 被拒绝：不在允许的目录内"
            )
        return abs_path

    def check_write(self, path: str) -> str:
        """验证写入权限。返回标准化路径或抛出 PermissionError。"""
        abs_path = os.path.abspath(path)
        if self.allowed_dir and not abs_path.startswith(self.allowed_dir):
            raise PermissionError(
                f"写入 '{path}' 被拒绝：不在允许的目录内"
            )
        filename = os.path.basename(abs_path)
        if filename.startswith(".env"):
            raise PermissionError(
                f"写入 '{filename}' 被拒绝：敏感文件"
            )
        return abs_path