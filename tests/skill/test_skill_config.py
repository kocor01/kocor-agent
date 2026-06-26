"""测试 Config 中的 Skill 相关配置。"""

import os
from pathlib import Path

from kocor.config import Config


class TestConfigSkillDefaults:
    """测试 Config 中 skill 字段的默认值"""

    def test_default_skills_config(self):
        cfg = Config()
        assert cfg.skills_config == "kocor.skills.json"

    def test_default_skills_dir(self):
        cfg = Config()
        assert cfg.skills_dir == "skills"

    def test_custom_skills_config(self):
        cfg = Config(skills_config="custom_skills.json")
        assert cfg.skills_config == "custom_skills.json"

    def test_custom_skills_dir(self):
        cfg = Config(skills_dir="my_skills")
        assert cfg.skills_dir == "my_skills"


class TestLoadConfigSkillEnv:
    """测试从环境变量加载 skill 配置"""

    def setup_method(self):
        self._saved = {}
        for key in ["KOCOR_SKILLS_CONFIG", "KOCOR_SKILLS_DIR"]:
            self._saved[key] = os.environ.pop(key, None)
        # 创建测试用的配置文件（文件存在性校验需要）
        self._test_files = ["my_skills.json", "a.json"]
        for f in self._test_files:
            Path(f).touch()

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
        for f in self._test_files:
            Path(f).unlink(missing_ok=True)

    def test_load_skills_config_default(self):
        cfg = Config._load()
        assert cfg.skills_config == "kocor.skills.json"

    def test_load_skills_config_from_env(self):
        os.environ["KOCOR_SKILLS_CONFIG"] = "my_skills.json"
        cfg = Config._load()
        assert cfg.skills_config == "my_skills.json"

    def test_load_skills_dir_default(self):
        cfg = Config._load()
        assert cfg.skills_dir == "skills"

    def test_load_skills_dir_from_env(self):
        os.environ["KOCOR_SKILLS_DIR"] = "custom_skills"
        cfg = Config._load()
        assert cfg.skills_dir == "custom_skills"

    def test_load_both_skill_env(self):
        os.environ["KOCOR_SKILLS_CONFIG"] = "a.json"
        os.environ["KOCOR_SKILLS_DIR"] = "b"
        cfg = Config._load()
        assert cfg.skills_config == "a.json"
        assert cfg.skills_dir == "b"