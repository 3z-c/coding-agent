"""agent 主循环：一轮 思考->行动->观察（ReAct）+ 循环终止条件。

设计要点（"循环终止条件"、"错误处理"与"对话历史/上下文管理"）：
1. 用户消息只在任务开始时追加一次，循环中不重复追加，避免污染历史
2. 工具执行失败以 ToolResult.error 回填给模型，让模型自我修正而不是被吞掉
3. 工具参数解析失败记录 parse_error 并回填错误，不静默丢弃
4. LLM 调用失败做指数退避重试，不直接崩溃
5. 上下文由 Memory 管理：工具结果分级（截断 / 落盘）+ token 预算裁剪
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from agent.memory import Memory
from agent.messages import Message
from llm.client import ChatResult, LLMClient
from tools.registry import ToolRegistry


class AgentError(Exception):
    """agent 运行期的可预期错误（达到步数上限、LLM 连续失败等）。"""


class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 20,
        max_retries: int = 3,
        verbose: bool = True,
        max_tokens: int = 20000,
        max_tool_result_chars: int = 8000,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.results_dir = results_dir
        self.memory: Memory | None = None
        self.total_tokens = 0

    @property
    def history(self) -> list[Message]:
        """当前对话历史（由 Memory 管理）。"""
        return self.memory.messages if self.memory else []

    #  终止条件：模型不再请求工具，或达到最大步数
    def run(self, task: str) -> str:
        """开启一个新会话并执行任务。返回最终回答文本。"""
        self._reset_session()   # 清空历史，新建 Memory
        self.memory.add_system(self.system_prompt)
        return self.continue_run(task)

    def continue_run(self, task: str) -> str:
        """在当前会话（保留已有历史）上追加一个新指令并继续执行。用于交互式多轮。

        与 run() 的区别：不清空 memory，agent 记得自己之前说过、做过什么。
        首次调用时（未经过 run()）也会自动初始化一个新会话。
        """
        if self.memory is None:
            self._reset_session()
            self.memory.add_system(self.system_prompt)
        self.memory.add_user(task)
        for step in range(self.max_steps):
            answer = self._step(step)
            if answer is not None:
                return answer

        raise AgentError(f"达到最大步数 {self.max_steps} 仍未完成，请简化任务或增大 --steps")

    def _reset_session(self) -> None:
        """清空会话：换新的 Memory 并归零 token 计数。"""
        self.memory = Memory(
            max_tokens=self.max_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            results_dir=self.results_dir,
        )
        self.total_tokens = 0

    def _step(self, step: int) -> Optional[str]:
        """执行一轮 思考->行动->观察。返回最终回答则结束，否则返回 None 继续循环。"""
        resp = self._call_llm()
        self.total_tokens += resp.total_tokens

        # 把模型这条 assistant 消息记入历史（含 tool_calls，供 API 校验上下文）
        self.memory.add_assistant(resp.content, resp.tool_calls if resp.has_tool_calls else None)

        # 终止条件 1：模型不再请求工具 -> 这就是最终回答
        if not resp.has_tool_calls:
            return resp.content if resp.content else "(模型没有返回任何内容)"

        # 终止条件 2：执行所有工具调用并回填结果
        for tc in resp.tool_calls:
            self._log(f">>> step {step} | 调用工具 {tc.name}({tc.arguments})")

            # 参数解析失败：不执行，把错误回填让模型修正
            if tc.parse_error:
                self.memory.add_parsing_error_tool(tc)
                self._log(f"    [ERR] 参数解析失败: {tc.parse_error}")
                continue

            result = self.registry.execute(tc.name, tc.arguments)
            first_line = result.output.splitlines()[0] if result.output else ""
            self._log(f"    [{'ERR' if result.is_error else 'OK'}] {first_line[:120]}")
            self.memory.add_tool(result.output, tool_call_id=tc.id, name=tc.name)

        # 上下文管理：每轮结束尝试裁剪
        removed = self.memory.trim()
        if removed:
            self._log(f"[上下文] 已裁剪 {removed} 条旧消息，避免超出上下文窗口")

        return None  # 继续循环

    # 错误处理：带指数退避重试的 LLM 调用
    def _call_llm(self) -> ChatResult:
        for attempt in range(self.max_retries):
            try:
                return self.client.chat(self.history, tools=self.registry.get_schemas())
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise AgentError(
                        f"LLM 调用连续失败 {self.max_retries} 次: {type(e).__name__}: {e}"
                    ) from e
                delay = 2**attempt
                self._log(f"[重试 {attempt + 1}] LLM 调用失败（{type(e).__name__}），{delay}s 后重试")
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text, file=sys.stderr)
