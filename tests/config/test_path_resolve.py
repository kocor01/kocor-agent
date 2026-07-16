"""Config _resolve_path 策略测试。"""

from __future__ import annotations

import os

from kocor.config import _resolve_path


class TestPathResolve:
    def test_absolute_path_returned_as_is(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text("{}")
        result = _resolve_path(str(target), prefer_cwd=True)
        assert result == str(target.resolve())

        result = _resolve_path(str(target), prefer_cwd=False)
        assert result == str(target.resolve())

    def test_config_prefers_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "kocor.mcp.json"
        cfg_file.write_text("{}")
        result = _resolve_path("kocor.mcp.json", prefer_cwd=True)
        # 文件存在时返回相对路径（匹配原始 _resolve_config_path 行为）
        assert result == "kocor.mcp.json"

    def test_data_defaults_to_package_root(self):
        result = _resolve_path(".kocor/memories", prefer_cwd=False)
        # 应包含 .kocor/memories 且路径指向 src/kocor 上层
        assert ".kocor" in result
        assert "memories" in result
        assert os.path.isabs(result)

    def test_nonexistent_config_falls_back_to_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = _resolve_path("nonexistent.json", prefer_cwd=True)
        # 不存在时返回原始路径
        assert result == "nonexistent.json"