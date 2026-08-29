"""工具抽象层。

- Tool         ：抽象基类，规定每个工具必须提供 name/description/parameters/execute
- BaseTool     ：带参数 schema 构建与取值辅助方法，写具体工具时更省事
- ToolResult   ：工具执行结果，成功/错误分离

约定（"工具的定义与本地执行"）：
- execute() 收到的是模型解析后的参数字典
- 一切异常/非法参数都返回 ToolResult.error(...)，绝不抛异常——错误要反馈给模型让它修正
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolResult:
    """工具执行结果：成功/错误分离。"""

    output: str
    is_error: bool = False

    @staticmethod
    def success(output: str) -> "ToolResult":
        return ToolResult(output=output, is_error=False)

    @staticmethod
    def error(message: str) -> "ToolResult":
        return ToolResult(output=message, is_error=True)


class Tool(ABC):
    """工具抽象基类。子类需实现 name / description / parameters / execute。"""

    #类属性
    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema，用于发给模型 + 本地校验

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """执行工具。参数非法等一律返回 ToolResult.error，不要抛异常。"""
        raise NotImplementedError

    def to_openai_schema(self) -> dict:
        """把工具定义转成 OpenAI 兼容格式（发给模型的那一份）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class BaseTool(Tool):
    """带参数构建/取值辅助方法的工具基类。"""

    # 参数 schema 构建
    @staticmethod
    def string_param(description: str, **extra: Any) -> dict:
        p: dict[str, Any] = {"type": "string", "description": description}
        p.update(extra)
        return p

    @staticmethod
    def int_param(description: str, **extra: Any) -> dict:
        p: dict[str, Any] = {"type": "integer", "description": description}
        p.update(extra)
        return p

    @staticmethod
    def bool_param(description: str, **extra: Any) -> dict:
        p: dict[str, Any] = {"type": "boolean", "description": description}
        p.update(extra)
        return p

    @staticmethod
    def enum_param(description: str, values: list[str]) -> dict:
        return {"type": "string", "description": description, "enum": values}

    @staticmethod
    def build_schema(properties: dict[str, dict], required: Optional[list[str]] = None) -> dict:
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    # 参数取值（容错：类型可能是 str/int/bool 混用）
    def get_string(self, arguments: dict, key: str, default: Optional[str] = None) -> Optional[str]:
        value = arguments.get(key)
        if value is None:
            return default
        return str(value)

    def get_int(self, arguments: dict, key: str, default: Optional[int] = None) -> Optional[int]:
        value = arguments.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, (str, float)):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        return default

    def get_bool(self, arguments: dict, key: str, default: bool = False) -> bool:
        value = arguments.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return default
