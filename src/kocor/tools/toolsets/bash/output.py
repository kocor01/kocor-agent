"""输出后处理：ANSI 转义序列剥离、敏感信息脱敏、输出截断。"""

import re

# ANSI 转义序列正则
_ANSI_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def strip_ansi(text: str) -> str:
    """去除 ANSI 转义序列（颜色、光标控制等）。"""
    return _ANSI_PATTERN.sub('', text)


def truncate_output(text: str, max_chars: int = 50_000) -> str:
    """截断输出：保留头 40% + 尾 60%。

    当输出超过 max_chars 时，中间用截断标记替换。
    """
    if not text or len(text) <= max_chars:
        return text
    head_len = int(max_chars * 0.4)
    tail_len = max_chars - head_len
    return text[:head_len] + "\n[... output truncated ...]\n" + text[-tail_len:]


def redact_sensitive(output: str) -> str:
    """遮盖敏感信息（API Key、Token 等）。

    覆盖模式：
    - sk-... 格式的 API Key
    - key=value 格式的敏感配置
    - 引号包围的 token/secret
    """
    # sk- 开头的 API Key（OpenAI、Anthropic 等，限制最大长度防贪婪匹配）
    output = re.sub(r'\bsk-[A-Za-z0-9]{16,64}', 'sk-****', output)
    # 引号内的 token 值（长度 >= 16）
    output = re.sub(r'(["\'])([A-Za-z0-9_\-]{16,})\1', r'\1****\1', output)
    # key=value 格式的敏感值（限制最大长度防止贪婪匹配）
    output = re.sub(
        r'(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*["\']?\s*([A-Za-z0-9_\-]{16,64})',
        r'\1=****',
        output,
    )
    return output


class OutputProcessor:
    """输出后处理管道。"""

    @staticmethod
    def process(output: str, max_chars: int = 50_000) -> str:
        """对输出执行后处理管道：

        1. ANSI 转义序列剥离
        2. 敏感信息脱敏
        3. 内容截断
        """
        output = strip_ansi(output)
        output = redact_sensitive(output)
        output = truncate_output(output, max_chars=max_chars)
        return output