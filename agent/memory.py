"""对话历史管理（上下文管理）。

两块核心策略（"对话历史与上下文管理"）：
1. 工具结果超长截断——防止一次大输出塞爆上下文
2. 历史超长裁剪——保留 system + 用户任务 + 最近对话，丢弃最旧的中间消息

关键细节：裁剪后的消息序列必须仍是一个合法边界，否则 OpenAI 兼容 API 会拒绝请求：
  - 末尾不能是孤立的 tool 消息（它前面必须跟过 assistant 的 tool_calls）
  - 末尾不能是仍带着未完成 tool_calls 的 assistant 消息
"""
from __future__ import annotations

from agent.messages import Message, Role, ToolCall


class Memory:
    def __init__(self, max_messages: int = 40, max_tool_result_chars: int = 8000) -> None:
        # 裁剪算法依赖"保留 system+user+最近 N 条"的结构，N 太小会出错，故设下限
        self.max_messages = max(max_messages, 4)
        self.max_tool_result_chars = max_tool_result_chars
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    # ---- 追加各种消息 ----
    def add_system(self, content: str) -> None:
        self._messages.append(Message.system(content))

    def add_user(self, content: str) -> None:
        self._messages.append(Message.user(content))

    def add_assistant(self, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> None:
        self._messages.append(Message.assistant(content, tool_calls))

    def add_tool(self, content: str, tool_call_id: str, name: str) -> None:
        self._messages.append(Message.tool(self._truncate(content), tool_call_id, name))

    def add_parsing_error_tool(self, tool_call: ToolCall) -> None:
        """把参数解析失败反馈给模型（错误回填，让模型自我修正）。"""
        content = (
            f"你上一条工具调用的参数不是合法 JSON，无法执行。\n"
            f"工具: {tool_call.name}\n"
            f"原始参数: {tool_call.arguments_raw!r}\n"
            f"解析错误: {tool_call.parse_error}\n"
            f"请用合法 JSON 格式重新发起这次调用。"
        )
        self._messages.append(Message.tool(content, tool_call.id, tool_call.name))

    # ---- 上下文管理：超长裁剪 ----
    def trim(self) -> int:
        """裁剪历史，返回被删除的消息数。保留最前 2 条（system + user 任务）和最近的消息。

        裁剪会破坏 assistant(tool_calls) 与 tool 的配对关系，因此裁完要清理：
        - 头部：保留段第一个消息若是孤立的 tool 消息（其 assistant 已被裁掉），丢弃
        - 尾部：循环弹出孤立的 tool 消息和带未完成 tool_calls 的 assistant 消息，
          直到末尾是一个合法边界（user / 不带 tool_calls 的 assistant / system）
        """
        if len(self._messages) <= self.max_messages:
            return 0
        kept = self._messages[:2] + self._messages[-(self.max_messages - 2):]
        removed = len(self._messages) - len(kept)
        self._messages = kept

        # 修复头部：system + user 之后不能是孤立的 tool 消息
        while len(self._messages) > 2 and self._messages[2].role == Role.TOOL:
            self._messages.pop(2)
            removed += 1

        # 修复尾部：直到末尾既不是 tool 消息、也不是带 tool_calls 的 assistant
        while self._messages:
            last = self._messages[-1]
            if last.role == Role.TOOL or last.tool_calls:
                self._messages.pop()
                removed += 1
            else:
                break
        return removed

    def _truncate(self, content: str) -> str:
        if len(content) <= self.max_tool_result_chars:
            return content
        return (
            content[: self.max_tool_result_chars]
            + f"\n...[结果已截断，共 {len(content)} 字符，仅保留前 {self.max_tool_result_chars} 字符]"
        )
