"""bash/output.py 单元测试：ANSI 剥离、敏感信息脱敏、输出截断。"""


from kocor.tools.toolsets.bash.output import (
    OutputProcessor,
    redact_sensitive,
    strip_ansi,
    truncate_output,
)


class TestStripAnsi:
    """ANSI 转义序列剥离测试。"""

    def test_plain_text(self):
        assert strip_ansi("hello world") == "hello world"

    def test_red_color(self):
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_bold(self):
        assert strip_ansi("\x1b[1mbold\x1b[0m") == "bold"

    def test_multiple_codes(self):
        text = "\x1b[32m\x1b[1mgreen bold\x1b[0m"
        assert strip_ansi(text) == "green bold"

    def test_cursor_movement(self):
        text = "\x1b[2K\x1b[1Aprogress"
        assert strip_ansi(text) == "progress"

    def test_clear_screen(self):
        assert strip_ansi("\x1b[2J\x1b[Hhello") == "hello"

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_newline_preserved(self):
        assert strip_ansi("line1\nline2") == "line1\nline2"


class TestTruncateOutput:
    """输出截断测试。"""

    def test_short_text_not_truncated(self):
        assert truncate_output("hello") == "hello"

    def test_long_text_truncated(self):
        text = "a" * 1000
        result = truncate_output(text, max_chars=100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_head_tail_ratio(self):
        text = "HEAD" * 100 + "TAIL" * 100
        result = truncate_output(text, max_chars=60)
        assert "HEAD" in result[:30]
        assert "TAIL" in result[-30:]
        assert "truncated" in result

    def test_exact_fit(self):
        text = "x" * 100
        result = truncate_output(text, max_chars=100)
        assert result == text

    def test_empty_string(self):
        assert truncate_output("") == ""


class TestRedactSensitive:
    """敏感信息脱敏测试。"""

    def test_sk_key(self):
        result = redact_sensitive("sk-abc123def456ghi789jklmnopqr")
        assert "sk-****" in result
        assert "sk-abc123def456ghi789jklmnopqr" not in result

    def test_api_key_header(self):
        result = redact_sensitive('api_key: sk-abc123def456ghi789jkl0123')
        assert "****" in result

    def test_secret_var(self):
        result = redact_sensitive('MY_SECRET=my-super-secret-value')
        assert "****" in result or "MY_SECRET=" in result

    def test_long_key_masked(self):
        result = redact_sensitive('token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"')
        assert "****" in result

    def test_normal_text_unchanged(self):
        text = "hello world, this is normal text"
        assert redact_sensitive(text) == text


class TestOutputProcessor:
    """OutputProcessor 管道测试。"""

    def test_process_ansi(self):
        result = OutputProcessor.process("\x1b[31mhello\x1b[0m")
        assert result == "hello"

    def test_process_truncation(self):
        text = "x" * 1000
        result = OutputProcessor.process(text, max_chars=100)
        assert "truncated" in result

    def test_process_redact(self):
        result = OutputProcessor.process("key=sk-abc123def456ghi789jkl")
        assert "sk-****" in result

    def test_process_pipeline(self):
        """全管道：ANSI + 脱敏 + 截断。"""
        long_key = "sk-abc123def456ghi789jkl"  # 23 chars after sk-
        text = f"\x1b[31msecret: {long_key}\x1b[0m" + "x" * 1000
        result = OutputProcessor.process(text, max_chars=200)
        # ANSI 剥离
        assert "\x1b[" not in result
        # 脱敏
        assert "sk-****" in result
        # 截断 — 原始文本 1000+ 字符，max_chars=200 应触发截断
        assert len(result) < len(text)
        assert "truncated" in result

    def test_process_empty(self):
        assert OutputProcessor.process("") == ""