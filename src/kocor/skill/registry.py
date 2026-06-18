"""Skill 注册与执行中心。"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Callable

from kocor.skill.models import (
    InvokeStrategy,
    SkillContext,
    SkillDefinition,
    SkillResult,
    SkillType,
)
from kocor.tool_registry import ToolRegistry


class SkillRegistry:
    """技能注册与执行中心。

    负责:
    - 注册、查询、列出技能
    - 从 JSON 配置文件加载技能
    - 从 skills/ 目录自动发现技能
    - 按名称执行技能
    - 将可 LLM 触发的技能暴露为 ToolRegistry 中的工具
    """

    def __init__(self, tool_registry: ToolRegistry | None = None):
        self._skills: dict[str, SkillDefinition] = {}
        self._tool_registry = tool_registry

    # -- 注册与查询 ---------------------------------------------------------------

    def register(self, skill: SkillDefinition) -> None:
        """注册一个技能定义。

        Args:
            skill: 技能定义

        Raises:
            ValueError: 同名技能已注册
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' is already registered")
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """按名称获取技能定义。"""
        return self._skills.get(name)

    def list_skills(
        self,
        category: str | None = None,
        enabled_only: bool = True,
    ) -> list[SkillDefinition]:
        """列出已注册的技能，可选择按分类和启用状态过滤。"""
        skills = list(self._skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        if category is not None:
            skills = [s for s in skills if s.category == category]
        return skills

    # -- 从配置文件加载 -----------------------------------------------------------

    def load_from_config(self, config_path: str) -> None:
        """从 JSON 配置文件加载技能。

        JSON 格式:
            {
              "skills": {
                "skill_name": {
                  "type": "prompt" | "code",
                  "invoke": "both" | "slash" | "llm",
                  "description": "...",
                  "prompt_template": "...",
                  "prompt_role": "system" | "user",
                  "module": "...",
                  "function": "...",
                  "parameters": {...},
                  "category": "...",
                  "enabled": true,
                  "version": "1.0.0",
                  "author": ""
                }
              }
            }
        """
        if not config_path or not os.path.exists(config_path):
            return

        import json

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_skills: dict = data.get("skills", {})
        for name, cfg in raw_skills.items():
            skill_type = SkillType(cfg.get("type", "prompt"))

            invoke_raw = cfg.get("invoke", "both")
            invoke_map = {
                "slash": InvokeStrategy.SLASH,
                "llm": InvokeStrategy.LLM,
                "both": InvokeStrategy.BOTH,
            }
            invoke_strategy = invoke_map.get(invoke_raw, InvokeStrategy.BOTH)

            handler = None
            parameters = None
            if skill_type == SkillType.CODE:
                module_name = cfg.get("module", "")
                function_name = cfg.get("function", "handler")
                if module_name:
                    handler = self._import_handler(module_name, function_name)
                parameters = cfg.get("parameters")

            skill = SkillDefinition(
                name=name,
                description=cfg.get("description", ""),
                skill_type=skill_type,
                invoke_strategy=invoke_strategy,
                prompt_template=cfg.get("prompt_template", ""),
                prompt_role=cfg.get("prompt_role", "user"),
                handler=handler,
                parameters=parameters,
                category=cfg.get("category", "general"),
                enabled=cfg.get("enabled", True),
                version=cfg.get("version", "1.0.0"),
                author=cfg.get("author", ""),
            )
            self.register(skill)

    # -- 内部辅助 -----------------------------------------------------------------

    @staticmethod
    def _import_handler(module_name: str, function_name: str) -> Callable:
        """动态导入模块并获取 handler 函数。"""
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ImportError(
                f"Cannot import module '{module_name}' for skill handler"
            )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        handler = getattr(mod, function_name, None)
        if handler is None:
            raise AttributeError(
                f"Module '{module_name}' has no '{function_name}' function"
            )
        if not callable(handler):
            raise TypeError(
                f"'{module_name}.{function_name}' is not callable"
            )
        return handler

    # -- Cline 格式技能发现 (SKILL.md + _meta.json) ---------------------------------

    def discover_cline_skills(self, directory: str) -> None:
        """扫描目录中的子目录，发现 Cline 格式 (SKILL.md) 的技能。

        每个子目录包含 SKILL.md 文件（含 YAML frontmatter），注册为 PROMPT 类型技能。
        """
        skills_dir = Path(directory)
        if not skills_dir.is_dir():
            return

        for subdir in sorted(skills_dir.iterdir()):
            if not subdir.is_dir():
                continue

            skill_md = subdir / "SKILL.md"
            if not skill_md.is_file():
                continue

            name, description, body = self._parse_skill_md(skill_md)
            if name is None:
                continue

            # 已注册的同名跳过
            if name in self._skills:
                continue

            skill = SkillDefinition(
                name=name,
                description=description or "",
                skill_type=SkillType.PROMPT,
                invoke_strategy=InvokeStrategy.BOTH,
                prompt_template=body,
                prompt_role="user",
                handler=None,
                parameters=None,
                category="cline",
                enabled=True,
                version="1.0.0",
                author="",
            )
            self.register(skill)

    @staticmethod
    def _parse_skill_md(path: Path) -> tuple[str | None, str, str]:
        """解析 SKILL.md，提取 name, description, body（不含 frontmatter）。

        Returns:
            (name, description, body) 三元组，无法解析时 name=None
        """
        try:
            text = path.read_text("utf-8")
        except Exception:
            return None, "", ""

        if not text.startswith("---"):
            return None, "", ""

        # 分离 frontmatter 和 body
        parts = text.split("---", 2)
        if len(parts) < 3:
            return None, "", ""

        frontmatter = parts[1]
        body = parts[2].strip()

        # 手动解析简单 YAML key-value（避免引入 PyYAML 依赖）
        name: str | None = None
        description = ""

        for line in frontmatter.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                name = line[len("name:"):].strip().strip("\"'")
            elif line.startswith("description:"):
                description = line[len("description:"):].strip().strip("\"'")

        return name, description, body

    # -- 从目录发现 ---------------------------------------------------------------

    def discover_skills(self, directory: str) -> None:
        """扫描目录中的 Python 文件自动发现技能。

        每个 .py 文件可通过模块级常量声明技能:
            NAME (str)           — 必填
            DESCRIPTION (str)    — 必填
            SKILL_TYPE (str)     — "prompt" | "code" (默认: "code")
            INVOKE_STRATEGY (str) — "both" | "slash" | "llm" (默认: "both")
            PROMPT_TEMPLATE (str) — prompt 技能模板
            PROMPT_ROLE (str)    — "user" | "system" (默认: "user")
            PARAMETERS (dict)    — JSON Schema
            CATEGORY (str)       — 分类
            ENABLED (bool)       — 是否启用
            VERSION (str)        — 版本
            AUTHOR (str)         — 作者
            handler 函数         — CODE 技能的可调用对象
        """
        skill_dir = Path(directory)
        if not skill_dir.is_dir():
            return

        sys.path.insert(0, str(skill_dir.parent))

        try:
            for pyfile in sorted(skill_dir.glob("*.py")):
                if pyfile.name.startswith("_"):
                    continue

                mod_name = f"_kocor_skill_{pyfile.stem}"
                spec = importlib.util.spec_from_file_location(mod_name, pyfile)
                if spec is None or spec.loader is None:
                    continue

                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                name = getattr(mod, "NAME", None)
                if name is None:
                    continue

                # 配置加载优先，目录发现的同名技能跳过
                if name in self._skills:
                    continue

                skill_type_str = getattr(mod, "SKILL_TYPE", "code")
                skill_type = SkillType(skill_type_str)

                invoke_raw = getattr(mod, "INVOKE_STRATEGY", "both")
                invoke_map = {
                    "slash": InvokeStrategy.SLASH,
                    "llm": InvokeStrategy.LLM,
                    "both": InvokeStrategy.BOTH,
                }
                invoke_strategy = invoke_map.get(invoke_raw, InvokeStrategy.BOTH)

                handler = getattr(mod, "handler", None) if skill_type == SkillType.CODE else None

                skill = SkillDefinition(
                    name=name,
                    description=getattr(mod, "DESCRIPTION", ""),
                    skill_type=skill_type,
                    invoke_strategy=invoke_strategy,
                    prompt_template=getattr(mod, "PROMPT_TEMPLATE", ""),
                    prompt_role=getattr(mod, "PROMPT_ROLE", "user"),
                    handler=handler,
                    parameters=getattr(mod, "PARAMETERS", None),
                    category=getattr(mod, "CATEGORY", "discovered"),
                    enabled=getattr(mod, "ENABLED", True),
                    version=getattr(mod, "VERSION", "1.0.0"),
                    author=getattr(mod, "AUTHOR", ""),
                )
                self.register(skill)
        finally:
            sys.path.pop(0)

    # -- 执行技能 -----------------------------------------------------------------

    def execute(self, name: str, context: SkillContext) -> SkillResult:
        """按名称执行技能，根据 skill_type 分派。

        Args:
            name: 技能名称
            context: 执行上下文

        Returns:
            执行结果
        """
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(
                content=f"Error: skill '{name}' not found",
                skill_name=name,
                success=False,
                error=f"Skill '{name}' not found",
            )
        if not skill.enabled:
            return SkillResult(
                content=f"Error: skill '{name}' is disabled",
                skill_name=name,
                success=False,
                error=f"Skill '{name}' is disabled",
            )

        try:
            if skill.skill_type == SkillType.CODE:
                return self._execute_code_skill(skill, context)
            else:
                return self._execute_prompt_skill(skill, context)
        except Exception as e:
            return SkillResult(
                content=f"Error executing skill '{name}': {e}",
                skill_name=name,
                success=False,
                error=str(e),
            )

    def _execute_prompt_skill(
        self, skill: SkillDefinition, context: SkillContext,
    ) -> SkillResult:
        """执行 PROMPT 类型技能：渲染模板并返回文本。

        渲染后的 prompt 由调用方（Agent）注入消息列表。
        """
        rendered = skill.prompt_template.format(
            user_input=context.user_input,
            skill_name=skill.name,
            **context.extra,
        )
        return SkillResult(
            content=rendered,
            skill_name=skill.name,
            success=True,
        )

    def _execute_code_skill(
        self, skill: SkillDefinition, context: SkillContext,
    ) -> SkillResult:
        """执行 CODE 类型技能：调用 handler 函数。"""
        if skill.handler is None:
            return SkillResult(
                content=f"Error: code skill '{skill.name}' has no handler",
                skill_name=skill.name,
                success=False,
                error="No handler",
            )

        sig = inspect.signature(skill.handler)
        kwargs = {}

        if any(p.name == "user_input" for p in sig.parameters.values()):
            kwargs["user_input"] = context.user_input
        if any(p.name == "tools" for p in sig.parameters.values()):
            kwargs["tools"] = context.tool_registry
        if any(p.name == "context" for p in sig.parameters.values()):
            kwargs["context"] = context

        result = skill.handler(**kwargs)
        return SkillResult(
            content=str(result),
            skill_name=skill.name,
            success=True,
        )

    # -- 暴露为 Tool --------------------------------------------------------------

    def register_as_tools(self, tool_registry: ToolRegistry | None = None) -> None:
        """将可 LLM 触发的技能注册为 ToolRegistry 中的工具。

        SLASH 类型的技能不会被注册为工具。
        LLM 和 BOTH 类型的技能会注册为 skill_<name>。
        """
        registry = tool_registry or self._tool_registry
        if registry is None:
            return

        for skill in self._skills.values():
            if skill.invoke_strategy == InvokeStrategy.SLASH:
                continue
            if not skill.enabled:
                continue

            registry.register(
                name=f"skill_{skill.name}",
                description=skill.description,
                parameters=self._build_tool_parameters(skill),
                handler=lambda user_input="", _s=skill: self._tool_wrapper(_s, user_input),
            )

    def _build_tool_parameters(self, skill: SkillDefinition) -> dict:
        """构建暴露为 tool 时的 JSON Schema 参数定义。"""
        if skill.parameters is not None:
            return skill.parameters
        return {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": f"Input for the skill: {skill.description}",
                },
            },
            "required": ["user_input"],
        }

    def _tool_wrapper(self, skill: SkillDefinition, user_input: str) -> str:
        """Tool handler wrapper：执行 skill 并返回结果内容。"""
        context = SkillContext(
            user_input=user_input,
            tool_registry=self._tool_registry,
        )
        result = self.execute(skill.name, context)
        return result.content


# -- 装饰器辅助 -----------------------------------------------------------------


def skill(
    name: str | None = None,
    description: str = "",
    invoke_strategy: str = "both",
    category: str = "general",
    version: str = "1.0.0",
    author: str = "",
):
    """装饰器：将函数注册为 CODE 类型技能。

    用法:
        @skill(name="greet", description="Greet someone")
        def greet_handler(user_input: str) -> str:
            return f"Hello, {user_input}!"

    未指定 name 时使用函数名，description 未指定时使用函数 docstring。
    """
    invoke_map = {
        "slash": InvokeStrategy.SLASH,
        "llm": InvokeStrategy.LLM,
        "both": InvokeStrategy.BOTH,
    }

    def decorator(func):
        skill_name = name or func.__name__
        skill_desc = description or (func.__doc__ or "").strip()

        sd = SkillDefinition(
            name=skill_name,
            description=skill_desc,
            skill_type=SkillType.CODE,
            invoke_strategy=invoke_map.get(invoke_strategy, InvokeStrategy.BOTH),
            handler=func,
            parameters=_params_from_signature(func),
            category=category,
            enabled=True,
            version=version,
            author=author,
        )
        func._skill_definition = sd
        return func

    return decorator


def _params_from_signature(func: Callable) -> dict:
    """从函数签名生成 JSON Schema 参数字典。"""
    sig = inspect.signature(func)
    properties = {}
    required = []

    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param_name in ("user_input", "tools", "context"):
            continue

        param_type = "string"
        if param.annotation is not inspect.Parameter.empty:
            type_name = getattr(param.annotation, "__name__", str(param.annotation))
            param_type = type_map.get(type_name, "string")

        prop = {"type": param_type, "description": f"Parameter {param_name}"}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default

        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }