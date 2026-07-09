"""测试二进制扩展名工具。"""

from kocor.tools.toolsets.file.binary_extensions import BINARY_EXTENSIONS, has_binary_extension


class TestBinaryExtensions:
    """测试 BINARY_EXTENSIONS 集合。"""

    def test_contains_common_binary_extensions(self):
        """包含常见二进制扩展名。"""
        assert ".png" in BINARY_EXTENSIONS
        assert ".jpg" in BINARY_EXTENSIONS
        assert ".jpeg" in BINARY_EXTENSIONS
        assert ".gif" in BINARY_EXTENSIONS
        assert ".ico" in BINARY_EXTENSIONS

    def test_contains_archive_extensions(self):
        """包含归档文件扩展名。"""
        assert ".zip" in BINARY_EXTENSIONS
        assert ".tar" in BINARY_EXTENSIONS
        assert ".gz" in BINARY_EXTENSIONS
        assert ".7z" in BINARY_EXTENSIONS

    def test_contains_executable_extensions(self):
        """包含可执行文件扩展名。"""
        assert ".exe" in BINARY_EXTENSIONS
        assert ".dll" in BINARY_EXTENSIONS
        assert ".so" in BINARY_EXTENSIONS
        assert ".dylib" in BINARY_EXTENSIONS

    def test_does_not_contain_text_extensions(self):
        """不包含文本文件扩展名。"""
        assert ".py" not in BINARY_EXTENSIONS
        assert ".txt" not in BINARY_EXTENSIONS
        assert ".md" not in BINARY_EXTENSIONS
        assert ".json" not in BINARY_EXTENSIONS
        assert ".yaml" not in BINARY_EXTENSIONS
        assert ".toml" not in BINARY_EXTENSIONS
        assert ".html" not in BINARY_EXTENSIONS
        assert ".css" not in BINARY_EXTENSIONS
        assert ".js" not in BINARY_EXTENSIONS
        assert ".ts" not in BINARY_EXTENSIONS


class TestHasBinaryExtension:
    """测试 has_binary_extension 函数。"""

    def test_png_returns_true(self):
        """.png 文件返回 True。"""
        assert has_binary_extension("image.png") is True

    def test_jpg_returns_true(self):
        """.jpg 文件返回 True。"""
        assert has_binary_extension("photo.jpg") is True

    def test_py_returns_false(self):
        """.py 文件返回 False。"""
        assert has_binary_extension("main.py") is False

    def test_txt_returns_false(self):
        """.txt 文件返回 False。"""
        assert has_binary_extension("readme.txt") is False

    def test_path_with_uppercase_extension(self):
        """大写扩展名应匹配（大小写不敏感）。"""
        assert has_binary_extension("image.PNG") is True
        assert has_binary_extension("archive.ZIP") is True

    def test_path_with_multiple_dots(self):
        """多段式文件名，取最后一个点后的扩展名。"""
        assert has_binary_extension("path/to/file.tar.gz") is True   # .gz 是二进制
        assert has_binary_extension("path.to.file.png") is True

    def test_no_extension_returns_false(self):
        """无扩展名返回 False。"""
        assert has_binary_extension("Makefile") is False
        assert has_binary_extension("LICENSE") is False

    def test_empty_path_returns_false(self):
        """空路径返回 False。"""
        assert has_binary_extension("") is False