"""测试模糊匹配模块。"""

from kocor.tools.toolsets.file.fuzzy_match import fuzzy_find_and_replace, match_strategies


class TestFuzzyMatch:
    """测试模糊匹配各种策略。"""

    def test_exact_match(self):
        """精确匹配。"""
        content = "def foo():\n    return 1"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "def foo():", "def bar():"
        )
        assert err is None
        assert count == 1
        assert strategy == "exact"
        assert "def bar():" in new_content

    def test_exact_match_no_occurrence(self):
        """精确匹配无匹配。"""
        content = "def foo():\n    return 1"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "def baz():", "def bar():"
        )
        assert err is not None

    def test_line_trimmed_strategy(self):
        """行去空白匹配。"""
        content = "def foo():\n    return 1"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "  def foo():", "def bar():"
        )
        assert err is None
        assert count == 1
        assert strategy == "line_trimmed"

    def test_whitespace_normalized_strategy(self):
        """空白折叠匹配。"""
        content = "def  foo():\n    return  1"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "def foo():", "def bar():"
        )
        assert err is None
        assert count == 1
        assert strategy in ("whitespace_normalized", "exact")

    def test_indentation_flexible_strategy(self):
        """忽略缩进匹配（首行缩进与 old_string 不同）。"""
        content = "    return x + 1\n    return x + 2"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "return x + 1", "return x + 3"
        )
        assert err is None
        assert count == 1
        # 由于 exact 策略用 substring 匹配，单行情况 exact 会先命中
        # 多行情况 line_trimmed 或 indentation_flexible 可能命中
        assert strategy is not None

    def test_deep_indentation_mismatch(self):
        """深层缩进差异需要 indentation_flexible 匹配。"""
        content = "    if x:\n        return 1\n    else:\n        return 2"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content,
            "if x:\n    return 1\nelse:\n    return 2",
            "if x:\n    return 10\nelse:\n    return 20",
        )
        # exact 不会匹配（缩进不同），需要 indentation_flexible 或 line_trimmed
        assert err is None
        assert count == 1
        assert strategy is not None

    def test_trimmed_boundary_strategy(self):
        """仅首尾行去空白。"""
        content = "def foo():\n    return 1\n    pass"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "  def foo():\n    return 1\n  pass", "def bar():\n    return 1\n    pass"
        )
        assert err is None
        assert count == 1
        assert strategy is not None

    def test_replace_all_multiple_occurrences(self):
        """替换全部。"""
        content = "x = 1\nx = 2\nx = 3"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "x = ", "y = ", replace_all=True
        )
        assert err is None
        assert count == 3

    def test_old_string_empty(self):
        """空 old_string 返回错误。"""
        content = "test"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "", "new"
        )
        assert err is not None

    def test_old_string_equals_new_string(self):
        """old 与 new 相同返回错误。"""
        content = "test"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "test", "test"
        )
        assert err is not None

    def test_multiline_content(self):
        """多行内容匹配。"""
        content = """def foo():
    print("hello")
    return 42"""
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content,
            'def foo():\n    print("hello")\n    return 42',
            'def bar():\n    print("world")\n    return 99',
        )
        assert err is None
        assert count == 1
        assert "def bar():" in new_content
        assert 'print("world")' in new_content
        assert "return 99" in new_content

    def test_content_unchanged_when_no_match(self):
        """无匹配时内容不变。"""
        content = "original content"
        new_content, count, strategy, err = fuzzy_find_and_replace(
            content, "something else", "replacement"
        )
        assert new_content == content
        assert count == 0
        assert err is not None


class TestMatchStrategies:
    """测试各个匹配策略函数。"""

    def test_exact_strategy(self):
        """exact 策略函数。"""
        result = match_strategies["exact"]("hello world", "hello world")
        assert result is True

    def test_exact_strategy_false(self):
        """exact 策略不匹配。"""
        result = match_strategies["exact"]("hello world", "Hello World")
        assert result is False

    def test_line_trimmed_strategy(self):
        """line_trimmed 策略。"""
        result = match_strategies["line_trimmed"](
            "  hello  \n  world  ",
            "hello\nworld",
        )
        assert result is True

    def test_whitespace_normalized(self):
        """whitespace_normalized 策略。"""
        result = match_strategies["whitespace_normalized"](
            "hello    world",
            "hello world",
        )
        assert result is True

    def test_indentation_flexible(self):
        """indentation_flexible 策略。"""
        result = match_strategies["indentation_flexible"](
            "    hello\n        world",
            "hello\nworld",
        )
        assert result is True

    def test_trimmed_boundary(self):
        """trimmed_boundary 策略。"""
        result = match_strategies["trimmed_boundary"](
            "  first line  \n  middle  \n  last line  ",
            "first line  \n  middle  \nlast line",
        )
        assert result is True