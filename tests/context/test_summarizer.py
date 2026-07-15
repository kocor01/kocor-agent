"""测试 HistorySummarizer。"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from kocor.context.types import SummaryNode
from kocor.context.summarizer import HistorySummarizer
from kocor.hook.base import HookPoint, HookContext, HookResult, HookAction
from kocor.llm_provider.message import FunctionCall, Message, ToolCall


class FakeLLMForSummary:
    """伪造的 LLM 客户端，始终返回预设的摘要文本。"""

    def __init__(self, summary_text: str = "这是对话摘要"):
        self.summary_text = summary_text
        self.call_count = 0
        self.last_messages = None

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        self.call_count += 1
        self.last_messages = messages
        return Message(role="assistant", content=self.summary_text)

    def stream(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        raise NotImplementedError


def _patch_llm(summary_text: str = "这是对话摘要"):
    """返回一个上下文管理器，将 LlmFactory.create 替换为 FakeLLMForSummary。"""
    return patch(
        "kocor.llm_provider.llm_factory.LlmFactory.create",
        return_value=FakeLLMForSummary(summary_text=summary_text),
    )


class TestHistorySummarizer:
    """测试 HistorySummarizer。"""

    def test_summarize_returns_summary_node(self):
        """summarize() 应返回 SummaryNode。"""
        with _patch_llm():
            summarizer = HistorySummarizer()
            msgs = [
                Message(role="user", content="你好"),
                Message(role="assistant", content="你好！"),
            ]
            node = summarizer.summarize(msgs)
        assert isinstance(node, SummaryNode)
        assert node.summary == "这是对话摘要"
        assert node.message_count == 2

    def test_summarize_calls_llm(self):
        """summarize() 应调用 LLM。"""
        fake = FakeLLMForSummary()
        with patch("kocor.llm_provider.llm_factory.LlmFactory.create", return_value=fake):
            summarizer = HistorySummarizer()
            summarizer.summarize([
                Message(role="user", content="问题1"),
                Message(role="assistant", content="回答1"),
            ])
        assert fake.call_count == 1

    def test_summarize_token_count_estimated(self):
        """摘要的 token_count 应被估算。"""
        with _patch_llm(summary_text="短摘要"):
            summarizer = HistorySummarizer()
            node = summarizer.summarize([
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ])
        assert node.token_count > 0

    def test_summarize_empty_list(self):
        """空消息列表应返回空摘要。"""
        with _patch_llm():
            summarizer = HistorySummarizer()
            node = summarizer.summarize([])
        assert node.summary == ""
        assert node.message_count == 0
        assert node.token_count == 0

    def test_summarize_captures_indexes(self):
        """摘要应记录原始消息索引。"""
        with _patch_llm():
            summarizer = HistorySummarizer()
            msgs = [Message(role="user", content=str(i)) for i in range(5)]
            node = summarizer.summarize(msgs, start_index=2, end_index=7)
        assert node.original_start == 2
        assert node.original_end == 7

    def test_summarize_formats_messages_with_tool_calls(self):
        """消息包含工具调用时应被正确格式化。"""
        with _patch_llm():
            summarizer = HistorySummarizer()
            msgs = [
                Message(role="assistant", content="", tool_calls=[
                    ToolCall(id="c1", function=FunctionCall(name="read_file", arguments='{"path":"a.txt"}')),
                ]),
                Message(role="tool", content="file content", tool_call_id="c1"),
            ]
            node = summarizer.summarize(msgs)
        assert node.message_count == 2
        assert node.summary != ""

    def test_custom_summarization_prompt(self):
        """自定义摘要 prompt。"""
        fake = FakeLLMForSummary()
        with patch("kocor.llm_provider.llm_factory.LlmFactory.create", return_value=fake):
            summarizer = HistorySummarizer()
            custom_prompt = "请用中文总结：{history_text}"
            summarizer.summarization_prompt = custom_prompt
            summarizer.summarize([
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ])
        # 验证自定义 prompt 被使用
        assert fake.last_messages is not None
        last_msg = fake.last_messages[0]
        assert "请用中文总结" in last_msg.content

    def test_summarizer_creates_llm_via_factory(self):
        """HistorySummarizer 内部通过 LlmFactory 创建 LLM 客户端。"""
        with _patch_llm():
            summarizer = HistorySummarizer()
        assert summarizer.llm is not None
        assert summarizer.llm.provider == "fake"

    def test_summarizer_hooks_fire(self):
        """PRE_SUMMARIZE 钩子被调用并收到 history_length（P0.2 回归）。"""
        captured_pre = []

        class CaptureHook:
            hook_point = HookPoint.PRE_SUMMARIZE
            def run(self, ctx: HookContext) -> HookResult:
                if ctx.extra.get("history_length") is not None:
                    captured_pre.append({
                        "history_length": ctx.extra["history_length"],
                        "message_count": ctx.extra.get("message_count"),
                    })
                return HookResult(action=HookAction.CONTINUE)

        hook_mgr = MagicMock()
        hook_mgr.run.side_effect = lambda point, ctx: [CaptureHook().run(ctx)]

        with _patch_llm():
            summarizer = HistorySummarizer(hook_manager=hook_mgr)
            summarizer.summarize([
                Message(role="user", content="hi"),
            ])

        assert len(captured_pre) == 1, f"got {captured_pre}"
        assert captured_pre[0]["history_length"] > 0

    def test_summarizer_null_hook_manager(self):
        """hook_manager=None 时 _run_hooks 不抛异常。"""
        with _patch_llm():
            summarizer = HistorySummarizer(hook_manager=None)
            result = summarizer.summarize([
                Message(role="user", content="hi"),
            ])
        assert result is not None

    # ── P1.4 修复 ──

    def test_summarizer_with_mock_llm(self):
        """直接注入 MockLLM，不通过 LlmFactory 创建。"""
        fake_llm = FakeLLMForSummary(summary_text="直接注入的摘要")
        summarizer = HistorySummarizer(llm=fake_llm)
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
        ]
        node = summarizer.summarize(msgs)
        assert isinstance(node, SummaryNode)
        assert node.summary == "直接注入的摘要"
        assert fake_llm.call_count == 1

    def test_summarizer_null_event_emitter(self):
        """event_emitter=None 时 _emit_event 不抛异常。"""
        with _patch_llm():
            summarizer = HistorySummarizer(event_emitter=None)
            result = summarizer.summarize([
                Message(role="user", content="hi"),
            ])
        assert result is not None