"""LLM 客户端封装。

- chat() 是项目里唯一发消息的出口
- 内部负责：消息格式转换、工具定义注入、响应解析（tool_calls / finish_reason / token 用量）

约定：整个项目只有本文件直接依赖 openai SDK，其余模块只与 Message / ChatResult 打交道。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from agent.messages import Message, ToolCall
from config import Config


@dataclass
class ChatResult:
    """一次 LLM 调用的结构化结果。"""

    content: Optional[str]
    finish_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> ChatResult:
        """发消息并返回结构化结果。

        messages: 对话历史（Message 列表）
        tools:    OpenAI 格式的工具 schema 列表（由 tools 模块生成）
        """
        api_messages = [m.to_api_dict() for m in messages]

        kwargs: dict = {
            "model": self.config.model,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self.client.chat.completions.create(**kwargs)

        if not resp.choices:
            # 防御：SDK 正常情况必有 choices，这里兜底交给上层重试
            raise RuntimeError("LLM 返回了空的 choices，无法解析")
        choice = resp.choices[0]
        msg = choice.message

        # 解析 tool_calls：arguments 是字符串，需要 json.loads
        # 解析失败不丢弃：记录 parse_error，由 loop 反馈给模型让它自我修正
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or ""
            parsed: dict = {}
            parse_error: Optional[str] = None
            if raw.strip():
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("arguments 不是合法的 JSON 对象")
                except (json.JSONDecodeError, ValueError) as e:
                    parsed = {}
                    parse_error = str(e)
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments_raw=raw,
                    arguments=parsed,
                    parse_error=parse_error,
                )
            )

        #组装并返回 ChatResult
        usage = resp.usage
        return ChatResult(
            content=msg.content,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
