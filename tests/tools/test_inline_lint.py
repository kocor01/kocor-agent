"""测试内联 Lint 模块。"""

from kocor.tools.toolset.inline_lint import lint_content


class TestPythonLint:
    def test_valid_python(self):
        result = lint_content("test.py", "def foo():\n    return 1\n")
        assert result["status"] == "ok"

    def test_syntax_error(self):
        result = lint_content("test.py", "def foo(:\n    return 1\n")
        assert result["status"] == "error"

    def test_indentation_error(self):
        result = lint_content("test.py", "def foo():\nreturn 1\n")
        assert result["status"] == "error"

    def test_empty_file(self):
        result = lint_content("test.py", "")
        assert result["status"] == "ok"


class TestJsonLint:
    def test_valid_json(self):
        result = lint_content("test.json", '{"a": 1, "b": 2}')
        assert result["status"] == "ok"

    def test_invalid_json(self):
        result = lint_content("test.json", '{"a": 1, "b": 2,}')
        assert result["status"] == "error"

    def test_empty_json(self):
        result = lint_content("test.json", "")
        assert result["status"] == "ok"


class TestYamlLint:
    def test_valid_yaml(self):
        result = lint_content("test.yaml", "key: value\nlist:\n  - item1\n")
        assert result["status"] == "ok"

    def test_invalid_yaml(self):
        result = lint_content("test.yaml", "value\n  subkey: bad")
        assert result["status"] == "error"

    def test_empty_yaml(self):
        result = lint_content("test.yaml", "")
        assert result["status"] == "ok"


class TestTomlLint:
    def test_valid_toml(self):
        result = lint_content("test.toml", 'key = "value"\n[section]\nname = "test"\n')
        assert result["status"] == "ok"

    def test_invalid_toml(self):
        result = lint_content("test.toml", "key = \n")
        assert result["status"] == "error"

    def test_empty_toml(self):
        result = lint_content("test.toml", "")
        assert result["status"] == "ok"


class TestUnknownExtension:
    def test_unknown_skipped(self):
        result = lint_content("readme.md", "# Hello")
        assert result["status"] == "skipped"

    def test_no_extension_skipped(self):
        result = lint_content("Makefile", "all:\n\ttest\n")
        assert result["status"] == "skipped"