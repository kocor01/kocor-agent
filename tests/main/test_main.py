"""测试 CLI 入口"""

from unittest.mock import MagicMock, patch

from kocor.message import FunctionCall, StreamChunk, ToolCall


class TestCLIParseArgs:
    """测试 CLI 参数解析"""

    @patch("kocor.__main__.argparse.ArgumentParser")
    def test_parse_stream_flag(self, mock_parser_cls):
        """测试 --stream 参数解析"""
        from kocor.__main__ import parse_args

        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(stream=True, user_input=["hello"])

        stream_enabled, user_args = parse_args()
        assert stream_enabled is True
        assert user_args == ["hello"]

    @patch("kocor.__main__.argparse.ArgumentParser")
    def test_parse_no_stream(self, mock_parser_cls):
        """测试不带 --stream 参数"""
        from kocor.__main__ import parse_args

        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(stream=False, user_input=["hello"])

        stream_enabled, user_args = parse_args()
        assert stream_enabled is False


class TestCLIMain:
    """测试 CLI main 函数"""

    @patch("kocor.__main__.load_dotenv")
    @patch("kocor.__main__.load_config")
    @patch("kocor.__main__.create_llm_client")
    @patch("kocor.__main__.ToolRegistry")
    @patch("kocor.__main__.register_mcp_tools")
    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "你好"])
    def test_main_with_stream(self, mock_agent_cls, mock_mcp, mock_tools,
                              mock_llm, mock_config, mock_dotenv):
        """测试 --stream 模式"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="你"),
            StreamChunk(content="好"),
            StreamChunk(is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        with patch("sys.stdout", new_callable=MagicMock):
            main()

        mock_agent.stream.assert_called_once()
        mock_agent.run.assert_not_called()

    @patch("kocor.__main__.load_dotenv")
    @patch("kocor.__main__.load_config")
    @patch("kocor.__main__.create_llm_client")
    @patch("kocor.__main__.ToolRegistry")
    @patch("kocor.__main__.register_mcp_tools")
    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "你好"])
    def test_main_without_stream(self, mock_agent_cls, mock_mcp, mock_tools,
                                 mock_llm, mock_config, mock_dotenv):
        """测试非流式模式"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.run.return_value = "非流式结果"
        mock_agent_cls.return_value = mock_agent

        with patch("sys.stdout", new_callable=MagicMock):
            main()

        mock_agent.run.assert_called_once()
        mock_agent.stream.assert_not_called()

    @patch("kocor.__main__.load_dotenv")
    @patch("kocor.__main__.load_config")
    @patch("kocor.__main__.create_llm_client")
    @patch("kocor.__main__.ToolRegistry")
    @patch("kocor.__main__.register_mcp_tools")
    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor"])
    def test_main_no_input(self, mock_agent_cls, mock_mcp, mock_tools,
                           mock_llm, mock_config, mock_dotenv):
        """测试无输入时打印用法"""
        from kocor.__main__ import main

        mock_agent_cls.return_value = MagicMock()

        with patch("sys.stdout", new_callable=MagicMock), \
             patch("sys.stdin", MagicMock(read=lambda: "")), \
             patch("sys.exit") as mock_exit:
            main()

        assert mock_exit.called
        mock_exit.assert_called_with(1)

    @patch("kocor.__main__.load_dotenv")
    @patch("kocor.__main__.load_config")
    @patch("kocor.__main__.create_llm_client")
    @patch("kocor.__main__.ToolRegistry")
    @patch("kocor.__main__.register_mcp_tools")
    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "你好"])
    def test_stream_prints_tool_calls(self, mock_agent_cls, mock_mcp, mock_tools,
                                      mock_llm, mock_config, mock_dotenv):
        """测试流式模式下工具调用输出"""
        from kocor.__main__ import main

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
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "工具调用" in output
        assert "read_file" in output

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "你好"])
    def test_stream_tool_call_then_final_answer(self, mock_agent_cls):
        """测试工具调用后继续输出最终答案"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            # 第一轮: 文本 + 工具调用
            StreamChunk(content="我来"),
            StreamChunk(content="读文件"),
            StreamChunk(
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
                is_final=True,
            ),
            # 第二轮: 最终答案（无工具调用）
            StreamChunk(content="文件内容是: hello"),
            StreamChunk(is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        # 工具调用只打印一次（去重）
        assert output.count("• read_file") == 1
        assert "read_file" in output
        # 最终答案也输出了
        assert "文件内容是: hello" in output


class TestCLIReasoning:
    """测试 CLI 流式模式下 reasoning 输出"""

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "你好"])
    def test_stream_prints_reasoning(self, mock_agent_cls):
        """测试流式模式下 reasoning 内容输出"""
        from kocor.__main__ import main

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
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "让我" in output
        assert "思考一下" in output
        assert "答案是" in output
        assert "42" in output
        assert "首先我需要分析问题..." in output


class TestCLIFormattedOutput:
    """测试 CLI 格式化输出"""

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "读文件"])
    def test_stream_prints_tool_result(self, mock_agent_cls):
        """测试工具结果格式化输出"""
        from kocor.__main__ import main
        from kocor.message import ToolResult

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            # 第 1 轮: 工具调用
            StreamChunk(tool_calls=[ToolCall(
                id="call_1",
                function=FunctionCall(name="read_file", arguments='{"path": ".env"}'),
            )], is_final=True),
            # 工具结果（Agent 内部 yield）
            StreamChunk(tool_result=ToolResult(
                tool_call_id="call_1",
                content="KOCOR_PROVIDER=anthropic\nOPENAI_API_KEY=xxx",
            ), is_final=True),
            # 第 2 轮: 最终答案
            StreamChunk(content="文件内容是: KOCOR_PROVIDER=anthropic", is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        # 工具结果现在在"🔧 工具调用"区块下方
        assert "工具调用" in output
        assert "KOCOR_PROVIDER=anthropic" in output
        assert "read_file" in output

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "读文件"])
    def test_stream_round_header(self, mock_agent_cls):
        """测试轮次标题格式"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="答案是42", is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "第 1 次请求" in output
        assert "──" in output

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "读文件"])
    def test_stream_reasoning_section(self, mock_agent_cls):
        """测试思维链区块格式"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(reasoning="让我思考"),
            StreamChunk(content="答案是42", is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "思维过程" in output
        assert "让我思考" in output

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "读文件"])
    def test_stream_content_section(self, mock_agent_cls):
        """测试结果输出区块格式"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            StreamChunk(content="答案是42", is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "回答" in output
        assert "答案是42" in output

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "读文件"])
    def test_stream_tool_call_section(self, mock_agent_cls):
        """测试工具调用区块格式"""
        from kocor.__main__ import main

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
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "工具调用" in output
        assert "read_file" in output

    @patch("kocor.__main__.Agent")
    @patch("sys.argv", ["kocor", "--stream", "读文件"])
    def test_stream_multiple_rounds(self, mock_agent_cls):
        """测试多轮请求标题"""
        from kocor.__main__ import main

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            # 第 1 轮: 工具调用
            StreamChunk(content="我来"),
            StreamChunk(
                tool_calls=[ToolCall(
                    id="call_1",
                    function=FunctionCall(name="read_file", arguments='{"path": "a.txt"}'),
                )],
                is_final=True,
            ),
            # 第 2 轮: 最终答案
            StreamChunk(content="文件内容是: hello"),
            StreamChunk(is_final=True),
        ])
        mock_agent_cls.return_value = mock_agent

        captured = []

        def fake_print(*args, **kwargs):
            captured.append("".join(str(a) for a in args))

        with patch("kocor.__main__.print", side_effect=fake_print):
            main()

        output = "".join(captured)
        assert "第 1 次请求" in output
        assert "第 2 次请求" in output
