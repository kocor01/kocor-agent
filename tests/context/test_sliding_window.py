"""测试 SlidingWindowStrategy。"""

from __future__ import annotations

import os

from kocor.config import Config
from kocor.context.types import SummaryNode
from kocor.context.sliding_window import SlidingWindowStrategy
from kocor.context.summarizer import HistorySummarizer
from kocor.llm_provider.message import FunctionCall, Message, ToolCall


class FakeLLMForSummary:
    """伪造的 LLM 客户端，返回预设摘要。"""

    def __init__(self, summary_text: str = "这是对话摘要"):
        self.summary_text = summary_text
        self.call_count = 0

    @property
    def provider(self):
        return "fake"

    def generate(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        self.call_count += 1
        return Message(role="assistant", content=self.summary_text)


def make_round(user_msg: str, assistant_reply: str,
               tool_calls: list[tuple[str, str]] | None = None) -> list[Message]:
    """构造一轮对话消息。"""
    msgs = [Message(role="user", content=user_msg)]

    if tool_calls:
        tc_list = [
            ToolCall(id=f"call_{i}", function=FunctionCall(name=name, arguments=args))
            for i, (name, args) in enumerate(tool_calls)
        ]
        msgs.append(Message(role="assistant", content="", tool_calls=tc_list))
        for tc in tc_list:
            msgs.append(Message(role="tool", content=f"result_{tc.id}", tool_call_id=tc.id))

    msgs.append(Message(role="assistant", content=assistant_reply))
    return msgs


class TestSlidingWindowStrategy:
    """测试 SlidingWindowStrategy。"""

    def setup_method(self):
        self.summarizer = HistorySummarizer(llm=FakeLLMForSummary())
        self.strategy = SlidingWindowStrategy(summarizer=self.summarizer, preserve_last_rounds=2, preserve_first_rounds=0)

    def test_no_truncation_needed(self):
        """历史消息少于保留轮次时，不截断。"""
        msgs = make_round("你好", "你好！")
        result, summary = self.strategy.apply(msgs, current_usage=100)
        assert summary is None
        assert len(result) == 2  # user + assistant

    def test_truncates_old_rounds(self):
        """超出保留轮次时，旧轮次被摘要。"""
        all_msgs = []
        for i in range(5):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = self.strategy.apply(all_msgs, current_usage=100)
        # 应保留最近 2 轮（4 条消息） + 摘要
        assert summary is not None
        assert len(result) <= 6  # 2 rounds × 2 + margin

    def test_preserves_latest_rounds(self):
        """最近的轮次应完整保留。"""
        all_msgs = []
        for i in range(4):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = self.strategy.apply(all_msgs, current_usage=100)
        # 最新一轮的内容应保留
        last_msgs_text = [m.content for m in result]
        assert "回答3" in " ".join(last_msgs_text)
        assert "问题3" in " ".join(last_msgs_text)

    def test_aggressive_mode(self):
        """AGGRESSIVE 策略只保留最后一轮。"""
        strategy = SlidingWindowStrategy(summarizer=self.summarizer, preserve_last_rounds=1, preserve_first_rounds=0)
        all_msgs = []
        for i in range(4):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = strategy.apply(all_msgs, current_usage=100)
        assert summary is not None
        assert len(result) <= 3  # 1 round × 2 + margin
        assert "问题3" in " ".join(m.content for m in result)

    def test_limited_tokens_triggers_aggressive(self):
        """token 空间不足时自动降级。"""
        old = os.environ.get("KOCOR_CONTEXT_MAX_TOKENS")
        os.environ["KOCOR_CONTEXT_MAX_TOKENS"] = "50"
        Config.reset()
        try:
            strategy = SlidingWindowStrategy(summarizer=self.summarizer, preserve_last_rounds=3, preserve_first_rounds=0)
            all_msgs = []
            for i in range(3):
                all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

            # 极小 max_tokens 应触发紧急截断
            result, summary = strategy.apply(all_msgs, current_usage=40)
            assert summary is not None
            assert len(result) <= len(all_msgs)
        finally:
            if old is None:
                del os.environ["KOCOR_CONTEXT_MAX_TOKENS"]
            else:
                os.environ["KOCOR_CONTEXT_MAX_TOKENS"] = old
            Config.reset()

    def test_empty_messages(self):
        """空消息列表应返回空。"""
        result, summary = self.strategy.apply([], current_usage=0)
        assert result == []
        assert summary is None

    def test_single_round(self):
        """只有一轮时不截断。"""
        msgs = make_round("问题", "回答")
        result, summary = self.strategy.apply(msgs, current_usage=0)
        assert summary is None
        assert len(result) == 2

    def test_round_with_tool_calls(self):
        """包含工具调用的轮次应整体保留。"""
        all_msgs = []
        for i in range(3):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}",
                                       tool_calls=[("read_file", '{"path": "a.txt"}')]))

        result, summary = self.strategy.apply(all_msgs, current_usage=0)
        assert summary is not None
        # 最近的轮次应包含工具调用信息
        last_round = [m for m in result if m.role == "assistant" and m.tool_calls]
        assert len(last_round) <= 2  # 保留轮次内的工具调用

    # ── 三段落策略（preserve_first_rounds） ──────────────────────

    def test_preserve_first_rounds_keeps_initial_rounds(self):
        """preserve_first_rounds>0 时最开始 N 轮应完整保留。"""
        strategy = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=2,
            preserve_first_rounds=2,
        )
        all_msgs = []
        for i in range(6):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = strategy.apply(all_msgs, current_usage=100)
        assert summary is not None
        result_text = " ".join(m.content or "" for m in result)
        # 最开始 2 轮应保留
        assert "问题0" in result_text
        assert "回答0" in result_text
        assert "问题1" in result_text
        # 最近 2 轮应保留
        assert "问题4" in result_text
        assert "问题5" in result_text
        # 摘要应嵌入在 first 和 last 之间
        summary_msgs = [m for m in result if m.role == "system"]
        assert len(summary_msgs) >= 1

    def test_first_plus_last_exceeds_total_no_summary(self):
        """preserve_first + preserve_last >= 总轮次时不应摘要。"""
        strategy = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=2,
            preserve_first_rounds=3,
        )
        all_msgs = []
        for i in range(4):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = strategy.apply(all_msgs, current_usage=100)
        assert summary is None  # 不应摘要
        assert len(result) == len(all_msgs)  # 全部保留

    def test_preserve_first_rounds_zero_backward_compat(self):
        """preserve_first_rounds=0 时行为与原来一致。"""
        strategy_zero = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=2,
            preserve_first_rounds=0,
        )
        strategy_default = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=2,
            preserve_first_rounds=0,
        )
        all_msgs = []
        for i in range(5):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result_zero, summary_zero = strategy_zero.apply(all_msgs, current_usage=100)
        result_default, summary_default = strategy_default.apply(all_msgs, current_usage=100)
        assert (summary_zero is None) == (summary_default is None)
        assert len(result_zero) == len(result_default)

    def test_preserve_first_middle_empty_no_summary(self):
        """first + last 刚好覆盖所有轮次，middle 为空时应无摘要。"""
        strategy = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=2,
            preserve_first_rounds=2,
        )
        all_msgs = []
        for i in range(4):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = strategy.apply(all_msgs, current_usage=100)
        assert summary is None
        assert len(result) == len(all_msgs)

    def test_preserve_first_keeps_correct_order(self):
        """三段式的消息顺序应为: first → summary → last。"""
        strategy = SlidingWindowStrategy(
            summarizer=self.summarizer,
            preserve_last_rounds=2,
            preserve_first_rounds=2,
        )
        all_msgs = []
        for i in range(6):
            all_msgs.extend(make_round(f"问题{i}", f"回答{i}"))

        result, summary = strategy.apply(all_msgs, current_usage=100)
        assert summary is not None
        # 找到摘要消息的位置
        summary_idx = None
        for i, m in enumerate(result):
            if m.role == "system":
                summary_idx = i
                break
        assert summary_idx is not None
        # first 段落应在摘要之前
        first_texts = " ".join(m.content or "" for m in result[:summary_idx])
        assert "问题0" in first_texts
        assert "问题1" in first_texts
        # last 段落应在摘要之后
        last_texts = " ".join(m.content or "" for m in result[summary_idx + 1:])
        assert "问题4" in last_texts
        assert "问题5" in last_texts
