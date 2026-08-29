"""对话历史管理（上下文管理）。

核心策略：
1. 工具结果分级：小结果原样保留；中结果截断加标记；超大结果全文落盘到磁盘，
   历史里只留预览 + 路径指针（不丢数据，模型要用时可 read_file 读回）。
2. token 预算裁剪：按字符数估算每条消息的 token（chars/4 兜底），总估算超出
   max_tokens 时，保留 system + 用户任务 + 最近对话，丢弃最旧的中间轮次。

关键细节：裁剪后的消息序列必须仍是一个合法边界，否则 OpenAI 兼容 API 会拒绝请求：
  - 末尾不能是孤立的 tool 消息（它前面必须跟过 assistant 的 tool_calls）
  - 末尾不能是仍带着未完成 tool_calls 的 assistant 消息
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from agent.messages import Message, Role, ToolCall

# 估算 token 的兜底比例：字符数 / 4 ≈ token 数（模型 tokenizer 的平均水平）
CHARS_PER_TOKEN = 4
# 落盘时保留在消息里的预览长度
SPILL_PREVIEW_CHARS = 2000


class Memory:
    def __init__(
        self,
        max_tokens: int = 20000,
        max_tool_result_chars: int = 8000,
        disk_threshold: int = 20000,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.disk_threshold = disk_threshold
        self.results_dir = Path(results_dir) if results_dir else None
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
        self._messages.append(
            Message.tool(self._bound_tool_result(content, tool_call_id), tool_call_id, name)
        )

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

    # ---- 工具结果分级：截断 / 落盘 ----
    def _bound_tool_result(self, content: str, tool_call_id: str) -> str:
        """工具结果分级处理：
        - 不超过 max_tool_result_chars：原样保留
        - 超过且未到落盘阈值：截断加标记
        - 超过落盘阈值且有 results_dir：全文落盘，消息里留预览 + 路径指针
        """
        if len(content) <= self.max_tool_result_chars:
            return content
        if self.results_dir is not None and len(content) > self.disk_threshold:
            try:
                self.results_dir.mkdir(parents=True, exist_ok=True)
                safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", tool_call_id) or "result"
                path = self.results_dir / f"{safe_id}.txt"
                path.write_text(content, encoding="utf-8")
                return (
                    f"[结果已落盘] 完整内容已写入 {path}（共 {len(content)} 字符），"
                    f"如需继续处理请用 read_file 读取该文件。\n\n"
                    f"预览：\n{content[:SPILL_PREVIEW_CHARS]}"
                )
            except OSError:
                pass  # 落盘失败：降级为截断
        return (
            content[: self.max_tool_result_chars]
            + f"\n...[结果已截断，共 {len(content)} 字符，仅保留前 {self.max_tool_result_chars} 字符]"
        )

    # ---- token 估算 ----
    @staticmethod
    def _estimate(msg: Message) -> int:
        """按字符数估算一条消息的 token 数（chars/4 兜底，至少 1）。"""
        return max(1, len(msg.content or "") // CHARS_PER_TOKEN)

    # ---- 上下文管理：token 预算裁剪 ----
    def trim(self) -> int:
        """裁剪历史，返回被删除的消息数。

        估算总 token 超出 max_tokens 时：保留 system + 用户任务（前 2 条）+
        最近对话，丢弃最旧的中间轮次。裁剪点向后对齐到"轮次起点"——即以
        assistant(tool_calls) 开头的消息，保证不拆散 tool 结果与其所属调用；
        末尾若残留未回填的 tool_calls 一并丢弃（API 要求 assistant 的 tool_calls
        必须紧跟对应 tool 消息）。
        """
        if len(self._messages) <= 4:
            return 0
        if sum(self._estimate(m) for m in self._messages) <= self.max_tokens:
            return 0

        # 从末尾往回累积最近对话的估算 token，超过预算的位置作为保留起点
        acc = 0
        cut = len(self._messages)
        for i in range(len(self._messages) - 1, 1, -1):  # 跳过 system + user 头
            acc += self._estimate(self._messages[i])
            if acc >= self.max_tokens:
                cut = i
                break
        if cut >= len(self._messages):
            return 0  # 尾部累积未达预算，溢出来自头两条本身，无可裁

        # 向后对齐到轮次起点（不拆 assistant(tool_calls) 与其后续 tool 消息）
        while cut > 2 and self._messages[cut].role == Role.TOOL:
            cut -= 1
        if cut <= 2:
            return 0  # 预算边界落在第一轮内部，无法安全裁剪

        kept = self._messages[:2] + self._messages[cut:]
        removed = len(self._messages) - len(kept)
        self._messages = kept

        # 修复尾部：末尾不能是带未完成 tool_calls 的 assistant 消息
        while self._messages and self._messages[-1].role == Role.ASSISTANT and self._messages[-1].tool_calls:
            self._messages.pop()
            removed += 1
        return removed
