"""二进制文件扩展名定义。

用于 read_file 工具在读取文件前快速判断文件类型，
避免尝试读取二进制文件浪费 token 或导致终端乱码。
"""

from __future__ import annotations

# 常见二进制文件扩展名集合（小写）
BINARY_EXTENSIONS: frozenset[str] = frozenset({
    # 图片
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg",
    ".ico", ".heic", ".heif", ".avif",
    # 字体
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # 音频/视频
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma",
    ".avi", ".mov", ".mkv", ".flv", ".webm",
    # 归档/压缩
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".zst",
    # 可执行/库
    ".exe", ".dll", ".so", ".dylib", ".bin", ".elf", ".o", ".a", ".lib",
    ".msi", ".apk", ".appimage", ".deb", ".rpm",
    # 文档（非纯文本）
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # 数据库/序列化
    ".db", ".sqlite", ".sqlite3", ".mdb",
    ".pkl", ".pickle", ".joblib",
    # 编译/字节码
    ".pyc", ".pyo", ".pyd", ".class", ".jar",
    ".wasm",
    # 其他
    ".iso", ".img", ".vmdk", ".qcow2",
    ".cache", ".ndjson",
})


def has_binary_extension(path: str) -> bool:
    """检查文件路径是否具有二进制扩展名。

    Args:
        path: 文件路径

    Returns:
        如果文件扩展名在已知二进制扩展名列表中返回 True
    """
    if not path:
        return False
    # 提取最后一个点后面的部分，转换为小写
    idx = path.rfind(".")
    if idx == -1:
        return False
    ext = path[idx:].lower()
    return ext in BINARY_EXTENSIONS