"""测试子代理系统提示构建与摘要截断。

TDD：先写测试验证纯函数行为，再确保实现匹配。
"""

from __future__ import annotations

import pytest

from kocor.tools.toolsets.subagent.system_prompt import build_subagent_system_prompt
from kocor.tools.toolsets.subagent.summary import truncate_summary, extract_summary


class TestBuildSubagentSystemPrompt:
    """测试 build_subagent_system_prompt 纯函数。"""

    def test_basic_goal_only(self):
        prompt = build_subagent_system_prompt(goal="搜索文件中的 bug")
        assert "搜索文件中的 bug" in prompt
        assert "【输出要求】" in prompt
        assert "【上下文】" not in prompt
        assert "【工作目录】" not in prompt
        assert "子代理委派" not in prompt
        assert prompt.startswith("你是一个聚焦的子代理")

    def test_context_included(self):
        prompt = build_subagent_system_prompt(goal="修复 bug", context="文件路径: /tmp/a.py")
        assert "【上下文】" in prompt
        assert "文件路径: /tmp/a.py" in prompt

    def test_workspace_included(self):
        prompt = build_subagent_system_prompt(goal="修复 bug", workspace="/home/user/project")
        assert "【工作目录】" in prompt
        assert "/home/user/project" in prompt

    def test_orchestrator_includes_guidance(self):
        prompt = build_subagent_system_prompt(goal="修复 bug", is_orchestrator=True, depth=0)
        assert "【子代理委派指导】" in prompt
        assert "深度为 0" in prompt
        assert "深度为 1" in prompt

    def test_orchestrator_depth_shown(self):
        prompt = build_subagent_system_prompt(goal="修复 bug", is_orchestrator=True, depth=2)
        assert "深度为 2" in prompt
        assert "深度为 3" in prompt

    def test_context_stripped_when_empty(self):
        prompt = build_subagent_system_prompt(goal="修复 bug", context="  ")
        assert "【上下文】" not in prompt

    def test_goal_whitespace_stripped(self):
        prompt = build_subagent_system_prompt(goal="  修复 bug  ")
        assert "修复 bug" in prompt
        assert "  修复 bug  " not in prompt

    def test_project_instructions_injected_when_exists(self):
        """KOCOR.md 存在时，项目指令层被注入到提示中。"""
        import os
        if not os.path.exists("KOCOR.md"):
            pytest.skip("KOCOR.md 不存在")
        prompt = build_subagent_system_prompt(goal="测试")
        assert "## 项目指令" in prompt


class TestTruncateSummary:
    """测试 truncate_summary 纯函数。"""

    def test_no_truncation_needed(self):
        text = "短文本"
        result = truncate_summary(text, 100)
        assert result == text

    def test_truncation_disabled(self):
        text = "很长" * 1000
        result = truncate_summary(text, 0)
        assert result == text

    def test_exact_boundary(self):
        text = "a" * 100
        result = truncate_summary(text, 100)
        assert result == text

    def test_truncation_head_tail_preserved(self):
        text = "【开头】这是前几行内容。\n" * 30 + "【中间】\n" + "【结尾】这是最后几行内容。\n" * 30
        # 目标：截断到约 200 字符，验证头尾都存活
        result = truncate_summary(text, 200)
        assert "【开头】" in result
        assert "【结尾】" in result
        assert "truncated" in result

    def test_truncation_marker_present(self):
        text = "行" * 500
        result = truncate_summary(text, 100)
        assert "[... " in result
        assert " characters truncated ...]" in result

    def test_single_line_truncation(self):
        text = "a" * 1000
        result = truncate_summary(text, 100)
        assert len(result) < 200
        assert "truncated" in result


class TestExtractSummary:
    """测试 extract_summary 结构化结果构建。"""

    def test_extract_basic(self):
        result = extract_summary("完成了任务A", max_chars=8000, status="completed")
        assert result["status"] == "completed"
        assert result["summary"] == "完成了任务A"

    def test_extract_none_text(self):
        result = extract_summary(None, max_chars=8000, status="completed")
        assert result["summary"] == ""

    def test_extract_budget_exhausted(self):
        result = extract_summary("部分完成", max_chars=8000, status="budget_exhausted")
        assert result["status"] == "budget_exhausted"

    def test_extract_truncates(self):
        text = "x" * 300
        result = extract_summary(text, max_chars=100, status="completed")
        assert len(result["summary"]) < 300
        assert "[... " in result["summary"]