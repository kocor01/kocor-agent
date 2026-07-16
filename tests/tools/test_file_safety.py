"""测试文件安全守卫模块。"""

import os
import tempfile

from kocor.tools.toolsets.file.file_safety import (
    _looks_like_read_file_line_numbered_content,
    check_sensitive_path,
    get_read_block_error,
    is_internal_file_tool_content,
    is_write_denied,
)


class TestCheckSensitivePath:
    """测试敏感路径检查。"""

    def test_etc_passwd_blocked(self):
        """/etc/passwd 被阻断。"""
        err = check_sensitive_path("/etc/passwd")
        assert err is not None
        assert "敏感" in err or "Refusing" in err

    def test_etc_shadow_blocked(self):
        """/etc/shadow 被阻断。"""
        err = check_sensitive_path("/etc/shadow")
        assert err is not None

    def test_boot_blocked(self):
        """/boot/ 路径被阻断。"""
        err = check_sensitive_path("/boot/vmlinuz")
        assert err is not None

    def test_regular_path_allowed(self):
        """普通项目路径放行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "main.py")
            err = check_sensitive_path(path)
            assert err is None

    def test_relative_path_to_etc_blocked(self):
        """相对路径指向 /etc/ 被阻断。"""
        # 模拟相对路径解析到 /etc/passwd 的情况
        path = "../etc/passwd"
        err = check_sensitive_path(path, allowed_dir="/tmp")
        # 相对路径 resolve 后可能变成 /etc/passwd
        assert err is not None

    def test_bin_path_blocked(self):
        """/bin/sh 等系统可执行目录被阻断。"""
        assert check_sensitive_path("/bin/sh") is not None

    def test_lib_path_blocked(self):
        """/lib/systemd 下的系统库目录被阻断。"""
        assert check_sensitive_path("/lib/systemd/evil") is not None

    def test_usr_path_blocked(self):
        """/usr/bin 下的路径被阻断。"""
        assert check_sensitive_path("/usr/bin/evil") is not None


class TestIsWriteDenied:
    """测试写拒绝列表检查。"""

    def test_ssh_key_denied(self):
        """SSH 私钥被拒绝写入。"""
        path = os.path.expanduser("~/.ssh/id_rsa")
        assert is_write_denied(path) is True

    def test_git_credentials_denied(self):
        """git-credentials 被拒绝写入。"""
        path = os.path.expanduser("~/.git-credentials")
        assert is_write_denied(path) is True

    def test_env_file_denied(self):
        """.env 文件被拒绝写入。"""
        path = os.path.join(os.getcwd(), ".env")
        assert is_write_denied(path) is True

    def test_regular_py_file_allowed(self):
        """普通的 .py 文件允许写入。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "main.py")
            assert is_write_denied(path) is False

    def test_aws_dir_denied(self):
        """~/.aws/ 目录下文件被拒绝写入。"""
        path = os.path.expanduser("~/.aws/credentials")
        assert is_write_denied(path) is True

    def test_kube_config_denied(self):
        """~/.kube/ 目录下文件被拒绝写入。"""
        path = os.path.expanduser("~/.kube/config")
        assert is_write_denied(path) is True

    def test_npmrc_denied(self):
        """~/.npmrc 被拒绝写入。"""
        path = os.path.expanduser("~/.npmrc")
        assert is_write_denied(path) is True


class TestGetReadBlockError:
    """测试读取阻断检查。"""

    def test_env_file_blocked(self):
        """读取 .env 文件被阻断。"""
        path = os.path.join(os.getcwd(), ".env")
        err = get_read_block_error(path)
        assert err is not None

    def test_regular_file_allowed(self):
        """普通 .py 文件放行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "main.py")
            open(py_file, "w").close()
            err = get_read_block_error(py_file)
            assert err is None

    def test_ssh_key_blocked(self):
        """读取 SSH 私钥被阻断。"""
        path = os.path.expanduser("~/.ssh/id_rsa")
        err = get_read_block_error(path)
        assert err is not None


class TestIsInternalFileToolContent:
    """测试内部工具内容检测。"""

    def test_dedup_status_message_detected(self):
        """去重状态消息被检测为内部内容。"""
        content = (
            "File unchanged since last read."
            " The content from the earlier read_file result in this conversation is still current"
        )
        assert is_internal_file_tool_content(content) is True

    def test_regular_code_not_detected(self):
        """普通代码不被检测为内部内容。"""
        content = "def hello():\n    print('world')"
        assert is_internal_file_tool_content(content) is False

    def test_empty_content_not_detected(self):
        """空内容不被检测。"""
        assert is_internal_file_tool_content("") is False
        assert is_internal_file_tool_content("   ") is False

    def test_short_text_with_status_message(self):
        """短文本包含状态消息被检测。"""
        from kocor.tools.toolsets.file.file_safety import _READ_DEDUP_STATUS_MESSAGE

        content = f"Note: {_READ_DEDUP_STATUS_MESSAGE}"
        assert is_internal_file_tool_content(content) is True

    def test_long_text_with_status_not_detected(self):
        """长文本即使包含状态消息也不被检测（内容大于 2 倍消息长度）。"""
        prefix = "File unchanged since last read. The content from the earlier read_file"
        suffix = "x" * 1000
        content = prefix + suffix
        assert is_internal_file_tool_content(content) is False


class TestLooksLikeReadFileContent:
    """测试 read_file 行号格式内容检测。"""

    def test_numbered_content_detected(self):
        """行号格式内容被检测。"""
        content = "1|import os\n2|import sys\n3|\n4|def main():\n5|    pass"
        assert _looks_like_read_file_line_numbered_content(content) is True

    def test_normal_content_not_detected(self):
        """普通内容不被检测。"""
        content = "import os\nimport sys\n\ndef main():\n    pass"
        assert _looks_like_read_file_line_numbered_content(content) is False

    def test_single_line_with_pipe_not_detected(self):
        """单行管道内容不被检测。"""
        content = "1|value"
        assert _looks_like_read_file_line_numbered_content(content) is False

    def test_few_numbered_lines_not_detected(self):
        """少于 2 行编号内容不被检测。"""
        content = "1|one line\nrandom text"
        assert _looks_like_read_file_line_numbered_content(content) is False
