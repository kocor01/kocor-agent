"""测试 skill() 装饰器。"""

from kocor.skill.models import InvokeStrategy, SkillType
from kocor.skill.skill_manager import _params_from_signature, skill


class TestSkillDecorator:
    """测试 @skill 装饰器"""

    def test_decorator_attaches_definition(self):
        @skill(name="greet", description="Greet someone")
        def greet_handler(user_input: str) -> str:
            return f"Hello, {user_input}!"

        assert hasattr(greet_handler, "_skill_definition")
        sd = greet_handler._skill_definition
        assert sd.name == "greet"
        assert sd.description == "Greet someone"
        assert sd.skill_type == SkillType.CODE
        assert sd.invoke_strategy == InvokeStrategy.BOTH
        assert sd.handler is greet_handler
        assert sd.enabled is True

    def test_decorator_uses_function_name(self):
        @skill()
        def my_handler(user_input: str) -> str:
            return f"Hi, {user_input}!"

        sd = my_handler._skill_definition
        assert sd.name == "my_handler"

    def test_decorator_uses_docstring_as_description(self):
        @skill()
        def greet(user_input: str) -> str:
            """Greet the user warmly."""
            return f"Hello, {user_input}!"

        sd = greet._skill_definition
        assert sd.description == "Greet the user warmly."

    def test_decorator_slash_only(self):
        @skill(name="deploy", description="Deploy", invoke_strategy="slash")
        def deploy_handler(env: str) -> str:
            return f"Deploying to {env}"

        sd = deploy_handler._skill_definition
        assert sd.invoke_strategy == InvokeStrategy.SLASH

    def test_decorator_llm_only(self):
        @skill(name="internal", description="Internal", invoke_strategy="llm")
        def internal_handler(data: str) -> str:
            return f"Processed: {data}"

        sd = internal_handler._skill_definition
        assert sd.invoke_strategy == InvokeStrategy.LLM

    def test_decorator_with_category(self):
        @skill(name="test", description="Test", category="testing")
        def test_handler(x: str) -> str:
            return x

        sd = test_handler._skill_definition
        assert sd.category == "testing"

    def test_decorator_function_can_be_called(self):
        @skill(name="greet", description="Greet")
        def greet_handler(user_input: str) -> str:
            return f"Hello, {user_input}!"

        result = greet_handler("world")
        assert result == "Hello, world!"

    def test_decorator_params_generated_from_signature(self):
        @skill(name="config", description="Config")
        def config_handler(env: str, port: int = 8080) -> str:
            return f"{env}:{port}"

        sd = config_handler._skill_definition
        assert sd.parameters is not None
        assert sd.parameters["type"] == "object"
        assert "env" in sd.parameters["properties"]
        assert "port" in sd.parameters["properties"]
        assert sd.parameters["required"] == ["env"]


class TestParamsFromSignature:
    """测试 _params_from_signature 辅助函数"""

    def test_simple_parameter(self):
        def func(name: str):
            pass

        params = _params_from_signature(func)
        assert params["type"] == "object"
        assert params["properties"]["name"]["type"] == "string"
        assert params["required"] == ["name"]

    def test_parameter_with_default(self):
        def func(name: str, age: int = 30):
            pass

        params = _params_from_signature(func)
        assert "name" in params["required"]
        assert "age" not in params["required"]
        assert params["properties"]["age"]["default"] == 30

    def test_skip_reserved_params(self):
        def func(user_input: str, tools: object, context: object, extra_arg: str):
            pass

        params = _params_from_signature(func)
        prop_names = list(params["properties"].keys())
        assert "user_input" not in prop_names
        assert "tools" not in prop_names
        assert "context" not in prop_names
        assert "extra_arg" in prop_names

    def test_type_mapping(self):
        def func(
            a: str,
            b: int,
            c: float,
            d: bool,
        ):
            pass

        params = _params_from_signature(func)
        assert params["properties"]["a"]["type"] == "string"
        assert params["properties"]["b"]["type"] == "integer"
        assert params["properties"]["c"]["type"] == "number"
        assert params["properties"]["d"]["type"] == "boolean"

    def test_no_params(self):
        def func() -> str:
            return "hello"

        params = _params_from_signature(func)
        assert params == {"type": "object", "properties": {}, "required": []}