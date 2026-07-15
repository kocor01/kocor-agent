"""测试 CLI 入口"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from kocor.llm_provider.message import FunctionCall, StreamChunk, ToolCall


def _make_mock_agent_builder(mock_agent: MagicMock) -> MagicMock:
    """创建链式调用的 mock AgentBuilder。"""
    mock_builder = MagicMock()
    mock_builder.build_llm.return_value = mock_builder
    mock_builder.build_subagent.return_value = mock_builder
    mock_builder.build_tools.return_value = mock_builder
    mock_builder.build_permission.return_value = mock_builder
    mock_builder.build_hooks.return_value = mock_builder
    mock_builder.build_session.return_value = mock_builder
    mock_builder.build.return_value = mock_agent
    return mock_builder


def _mock_cli_main_stack(argv: list[str], mock_agent: MagicMock | None = None) -> ExitStack:
    """创建 main() 所需的通用 mock 上下文栈。"""
    stack = ExitStack()
    mock_cfg = MagicMock(skills_config="", mcp_config="", max_iterations=20,
                          memory_dir="", context_strategy="default",
                          log_level="INFO", log_dir="./log")
    stack.enter_context(patch("kocor.config.Config.load", return_value=mock_cfg))
    stack.enter_context(patch("sys.argv", argv))

    if mock_agent is not None:
        mock_builder = _make_mock_agent_builder(mock_agent)
        stack.enter_context(patch("kocor.cli_builder.AgentBuilder", return_value=mock_builder))
    return stack


class TestCLIParseArgs:
    """测试 CLI 参数解析"""

    @patch("kocor.cli.argparse.ArgumentParser")
    def test_parse_default_stream(self, mock_parser_cls):
        """测试默认流式输出为 True"""
        from kocor.cli import parse_args

        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(
            stream=True, repl=False, user_input=["hello"],
        )

        args = parse_args()
        assert args.stream is True
        assert args.repl is False
        assert args.user_input == ["hello"]

    @patch("kocor.cli.argparse.ArgumentParser")
    def test_parse_no_stream_flag(self, mock_parser_cls):
        """测试 --no-stream 参数解析"""
        from kocor.cli import parse_args

        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(
            stream=False, repl=False, user_input=["hello"],
        )

        args = parse_args()
        assert args.stream is False
        assert args.repl is False


class TestCLIMain:
    """测试 CLI main 函数"""

    @patch("kocor.config.Config.load")
    @patch("kocor.cli_builder.AgentBuilder")
    @patch("sys.argv", ["kocor", "你好"])
    def test_main_with_stream(self, mock_builder_cls, mock_config):
        """测试默认流式输出模式"""
        from kocor.cli import main

        mock_config.return_value = MagicMock(skills_config="", max_iterations=20, log_level="INFO", log_dir="./log")
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="你"),
            StreamChunk(content="好"),
            StreamChunk(is_final=True),
        ])
        mock_builder = _make_mock_agent_builder(mock_agent)
        mock_builder_cls.return_value = mock_builder

        with patch("sys.stdout", new_callable=MagicMock):
            main()

        mock_agent.stream.assert_called_once()
        mock_agent.run.assert_not_called()

    @patch("kocor.config.Config.load")
    @patch("kocor.cli_builder.AgentBuilder")
    @patch("sys.argv", ["kocor", "--no-stream", "你好"])
    def test_main_without_stream(self, mock_builder_cls, mock_config):
        """测试 --no-stream 模式"""
        from kocor.cli import main

        mock_config.return_value = MagicMock(skills_config="", max_iterations=20, log_level="INFO", log_dir="./log")
        mock_agent = MagicMock()
        mock_agent.run.return_value = "非流式结果"
        mock_builder = _make_mock_agent_builder(mock_agent)
        mock_builder_cls.return_value = mock_builder

        with patch("sys.stdout", new_callable=MagicMock):
            main()

        mock_agent.run.assert_called_once()
        mock_agent.stream.assert_not_called()

    @patch("kocor.config.Config.load")
    @patch("kocor.cli_builder.AgentBuilder")
    @patch("sys.argv", ["kocor"])
    def test_main_no_input(self, mock_builder_cls, mock_config):
        """测试无输入时打印用法"""
        from kocor.cli import main

        mock_config.return_value = MagicMock(skills_config="", max_iterations=20, log_level="INFO", log_dir="./log")
        mock_agent = MagicMock()
        mock_builder = _make_mock_agent_builder(mock_agent)
        mock_builder_cls.return_value = mock_builder

        mock_stdin = MagicMock()
        mock_stdin.read.return_value = ""
        mock_stdin.isatty.return_value = False

        with patch("sys.stdout", new_callable=MagicMock), \
             patch("sys.stdin", mock_stdin), \
             patch("sys.exit") as mock_exit:
            main()

        assert mock_exit.called
        mock_exit.assert_called_with(1)

    @patch("kocor.config.Config.load")
    @patch("kocor.cli_builder.AgentBuilder")
    @patch("sys.argv", ["kocor", "你好"])
    def test_stream_prints_tool_calls(self, mock_builder_cls, mock_config):
        """测试流式模式下工具调用输出"""
        from kocor.cli import main

        mock_config.return_value = MagicMock(skills_config="", max_iterations=20, log_level="INFO", log_dir="./log")
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
                is_final=True,
            ),
        ])
        mock_builder = _make_mock_agent_builder(mock_agent)
        mock_builder_cls.return_value = mock_builder

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "工具调用" in output
        assert "read_file" in output

    def test_stream_tool_call_then_final_answer(self):
        """测试工具调用后继续输出最终答案"""
        from kocor.cli import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="我来"),
            StreamChunk(content="读文件"),
            StreamChunk(
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
                is_final=True,
            ),
            StreamChunk(content="文件内容是: hello"),
            StreamChunk(is_final=True),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "你好"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        # 工具调用只打印一次（去重）
        assert output.count("• read_file") == 1
        assert "read_file" in output
        # 最终答案也输出了
        assert "文件内容是: hello" in output


class TestCLIReasoning:
    """测试 CLI 流式模式下 reasoning 输出"""

    @patch("kocor.config.Config.load")
    @patch("kocor.cli_builder.AgentBuilder")
    @patch("sys.argv", ["kocor", "你好"])
    def test_stream_prints_reasoning(self, mock_builder_cls, mock_config):
        """测试流式模式下 reasoning 内容输出"""
        from kocor.cli import main

        mock_config.return_value = MagicMock(skills_config="", max_iterations=20, log_level="INFO", log_dir="./log")
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="让我"),
            StreamChunk(content="思考一下"),
            StreamChunk(
                content="答案是",
                reasoning="首先我需要分析问题...",
            ),
            StreamChunk(content="42", is_final=True),
        ])
        mock_builder = _make_mock_agent_builder(mock_agent)
        mock_builder_cls.return_value = mock_builder

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "让我" in output
        assert "思考一下" in output
        assert "答案是" in output
        assert "42" in output
        assert "首先我需要分析问题..." in output


_CLI_MAIN_MOCKS = [
    patch("kocor.config.Config.load"),
    patch("kocor.cli_builder.AgentBuilder"),
]


class TestCLIFormattedOutput:
    """测试 CLI 格式化输出"""

    def test_stream_prints_tool_result(self):
        """测试工具结果格式化输出"""
        from kocor.cli import main
        from kocor.llm_provider.message import ToolResult

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(tool_calls=[ToolCall(
                id="call_1",
                function=FunctionCall(name="read_file", arguments='{"path": ".env"}'),
            )], is_final=True),
            StreamChunk(tool_result=ToolResult(
                tool_call_id="call_1",
                content="KOCOR_PROVIDER=anthropic\nOPENAI_API_KEY=xxx",
            ), is_final=True),
            StreamChunk(content="文件内容是: KOCOR_PROVIDER=anthropic", is_final=True),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "读文件"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "工具调用" in output
        assert "KOCOR_PROVIDER=anthropic" in output
        assert "read_file" in output

    def test_stream_round_header(self):
        """测试轮次标题格式"""
        from kocor.cli import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="答案是42", is_final=True),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "读文件"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "第 1 次请求" in output
        assert "──" in output

    def test_stream_reasoning_section(self):
        """测试思维链区块格式"""
        from kocor.cli import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(reasoning="让我思考"),
            StreamChunk(content="答案是42", is_final=True),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "读文件"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "思维过程" in output
        assert "让我思考" in output

    def test_stream_content_section(self):
        """测试结果输出区块格式"""
        from kocor.cli import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="答案是42", is_final=True),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "读文件"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "回答" in output
        assert "答案是42" in output

    def test_stream_tool_call_section(self):
        """测试工具调用区块格式"""
        from kocor.cli import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
                is_final=True,
            ),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "读文件"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "工具调用" in output
        assert "read_file" in output

    def test_stream_multiple_rounds(self):
        """测试多轮请求标题"""
        from kocor.cli import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="我来"),
            StreamChunk(
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
                is_final=True,
            ),
            StreamChunk(content="文件内容是: hello"),
            StreamChunk(is_final=True),
        ])

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with _mock_cli_main_stack(["kocor", "读文件"], mock_agent), \
             patch("kocor.cli.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "第 1 次请求" in output
        assert "第 2 次请求" in output