"""测试输出截断模块"""

from kocor.mcp import TruncateConfig, truncate_output


class TestTruncateOutput:
    """测试输出截断功能"""

    def test_empty_string(self):
        """空字符串不变"""
        assert truncate_output("") == ""

    def test_short_text_unchanged(self):
        """短文本不截断"""
        text = "hello world"
        assert truncate_output(text) == text

    def test_line_truncation(self):
        """单行超长截断"""
        cfg = TruncateConfig(max_line_length=10, max_lines=100, max_bytes=100000)
        text = "a" * 20
        result = truncate_output(text, cfg)
        # 前 10 字符保留，后面被替换为截断标记
        assert result.startswith("a" * 10)
        assert "truncated" in result

    def test_line_count_truncation(self):
        """超出行数时头尾保留"""
        cfg = TruncateConfig(max_line_length=1000, max_lines=10, max_bytes=100000)
        lines = [f"line {i}" for i in range(20)]
        text = "\n".join(lines)
        result = truncate_output(text, cfg)
        # 头尾各 5 行，中间截断标记
        assert "line 0" in result
        assert "line 4" in result
        assert "truncated" in result
        assert "line 15" in result
        assert "line 19" in result

    def test_byte_truncation(self):
        """超字节时头尾截断"""
        cfg = TruncateConfig(max_line_length=1000, max_lines=1000, max_bytes=100)
        text = "a" * 200
        result = truncate_output(text, cfg)
        assert len(result) < 200
        assert "TRUNCATED" in result

    def test_default_config_used(self):
        """不传 config 时使用默认值"""
        text = "short"
        assert truncate_output(text) == text

    def test_large_text_truncated_to_default(self):
        """默认 50KB 限制生效"""
        text = "x" * 100_000
        result = truncate_output(text)
        assert len(result) < 100_000

    def test_exact_boundary_line_count(self):
        """行数刚好达到限制时不截断"""
        cfg = TruncateConfig(max_line_length=1000, max_lines=5, max_bytes=100000)
        text = "\n".join(f"line {i}" for i in range(5))
        assert truncate_output(text, cfg) == text

    def test_exact_boundary_line_length(self):
        """行长度刚好达到限制时不截断"""
        cfg = TruncateConfig(max_line_length=10, max_lines=100, max_bytes=100000)
        text = "a" * 10
        assert truncate_output(text, cfg) == text

    def test_mixed_content(self):
        """混合内容（短行 + 超长行）正确截断"""
        cfg = TruncateConfig(max_line_length=5, max_lines=100, max_bytes=100000)
        lines = ["short", "a" * 20, "ok"]
        text = "\n".join(lines)
        result = truncate_output(text, cfg)
        assert "short" in result
        assert "ok" in result
        assert "truncated" in result
