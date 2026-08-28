"""命令执行工具：run_command。

通过 subprocess 在本地 shell 执行命令。
- 必须设 timeout，防止命令挂死拖住整个 agent 循环
- 非零退出码不抛异常，作为 ToolResult.error 反馈给模型，让模型据此修复
"""
from __future__ import annotations

import subprocess

from tools.base import BaseTool, ToolResult

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_CHARS = 20000  # 超长输出截断，防止塞爆上下文


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[输出已截断，共 {len(text)} 字符，仅显示前 {limit} 字符]"


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "在本地 shell 中执行命令并返回输出。可用于运行测试、构建、查看进程、"
        "处理文件等。非零退出码会作为错误返回。"
    )

    parameters = BaseTool.build_schema(
        properties={
            "command": BaseTool.string_param("要执行的命令，例如：python -m pytest"),
            "timeout": BaseTool.int_param(f"超时秒数（默认 {_DEFAULT_TIMEOUT}，最大值 300）"),
        },
        required=["command"],
    )

    def execute(self, arguments: dict) -> ToolResult:
        command = self.get_string(arguments, "command")
        if not command:
            return ToolResult.error("缺少 command 参数")
        timeout = min(self.get_int(arguments, "timeout", _DEFAULT_TIMEOUT) or _DEFAULT_TIMEOUT, 300)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (e.stderr or "")
            return ToolResult.error(f"命令超时（>{timeout} 秒）: {command}\n{_truncate(str(partial))}")
        except OSError as e:
            return ToolResult.error(f"无法执行命令: {e}")

        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        output = _truncate(output)

        if proc.returncode != 0:
            # 非零退出码：不抛异常，反馈给模型修正
            return ToolResult.error(f"[exit code {proc.returncode}]\n{output}")
        return ToolResult.success(output if output else "(命令执行成功，无输出)")
