"""工具注册表。

- 把所有工具按 name 集中管理
- get_schemas() 输出 OpenAI 格式的工具列表（发给模型）
- execute() 是 agent 循环调用工具的唯一切口；查不到工具、工具内部异常都会被
  包装成 ToolResult.error 返回，绝不让循环崩溃
"""
from __future__ import annotations

from typing import Iterator, Optional

from tools.base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """OpenAI 格式的工具 schema 列表，发给模型用。"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"未找到工具: {name}")
        try:
            return tool.execute(arguments)
        except Exception as e:  # 工具内部崩溃也不打断循环
            return ToolResult.error(f"工具 {name} 执行异常: {type(e).__name__}: {e}")

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
