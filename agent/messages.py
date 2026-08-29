"""对话消息的数据模型。

- Role：system / user / assistant / tool 四角色
- Message：一条消息，按角色携带不同字段
- ToolCall：模型发起的一次工具调用（含原始参数、解析结果、解析错误）
- to_api_dict()：转成 OpenAI 兼容的 API 请求格式

字段约定：
  assistant 消息  -> 携带 tool_calls（模型要求调用的工具列表）
  tool 消息       -> 携带 tool_call_id（对应哪一次调用）+ name（工具名）+ content（工具结果）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


#定义四种消息角色
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments_raw: str = ""  # 模型返回的原始 arguments 字符串
    arguments: dict[str, Any] = field(default_factory=dict)  # 解析后的参数
    parse_error: Optional[str] = None  # arguments 解析失败时的错误信息（None 表示解析成功）

    def arguments_json(self) -> str:
        """转成 API 需要的 JSON 字符串：优先用原始串，保证与模型输出一致。"""
        if self.arguments_raw:
            return self.arguments_raw
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass
class Message:
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None  # assistant 消息：模型要求调用的工具
    tool_call_id: Optional[str] = None  # tool 消息：对应哪次调用
    name: Optional[str] = None  # tool 消息：工具名

    # ---- 工厂方法 ----
    @staticmethod
    def system(content: str) -> "Message":
        return Message(role=Role.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> "Message":
        return Message(role=Role.USER, content=content)

    @staticmethod
    def assistant(content: Optional[str] = None, tool_calls: Optional[list[ToolCall]] = None) -> "Message":
        return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @staticmethod
    def tool(content: str, tool_call_id: str, name: str) -> "Message":
        return Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)

    def to_api_dict(self) -> dict:
        """转成 OpenAI 兼容 API 的消息格式。"""
        msg: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls is not None:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json()},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg
