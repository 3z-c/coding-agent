"""编码 agent 命令行入口。

用法（项目根目录 coding-agent 下执行）:
    python main.py "你的任务"             # 单次任务，跑完即退
    python main.py                        # 交互模式：连续对话，Ctrl+C 或输入 exit 退出
    python main.py --steps 50             # 提高单次任务的步数上限
    python main.py --cwd ./my_project     # 指定 agent 的工作目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 提供命令行上下键历史（Unix；Windows 上不可用则忽略）
try:
    import readline  # noqa: F401
except ImportError:
    pass

from agent.loop import Agent, AgentError
from config import Config
from llm.client import LLMClient
from tools.edit import DeleteFileTool, EditFileTool
from tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from tools.registry import ToolRegistry
from tools.shell import RunCommandTool

SYSTEM_PROMPT = """你是一个运行在本地计算机上的编码助手（coding agent）。

你可以通过工具读写文件、执行命令，自主完成交给你的编程任务。规则：
- 动手前先想清楚，一次做一件明确的事
- 命令执行失败时，仔细阅读错误输出并自我修正，不要重复同样错误的命令
- 修改文件的某一段时，优先用 edit_file 精确替换，不要用 write_file 整文件重写
- 优先使用工作目录内相对路径
- 验证 GUI / 图形界面 / 会阻塞的程序时，不要尝试真正打开窗口或跑 mainloop()（会卡死或超时）。
  改为无头验证：创建对象、模拟点击/调用方法、打印状态、destroy，最后把真正打开窗口的命令留给用户自己运行
- 任务结束后用 delete_file 清理你创建的临时验证文件（如 _test.py），不要在工作目录留下垃圾
- 完成任务后，用中文简洁总结你做了什么、结果如何"""

EXIT_WORDS = {"exit", "quit", "q", "退出"}


def build_registry(workspace_root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(ReadFileTool(workspace_root=workspace_root))
    registry.add(WriteFileTool(workspace_root=workspace_root))
    registry.add(ListDirTool(workspace_root=workspace_root))
    registry.add(RunCommandTool())
    registry.add(EditFileTool(workspace_root=workspace_root))
    registry.add(DeleteFileTool(workspace_root=workspace_root))
    return registry


def run_interactive(agent: Agent) -> int:
    """交互模式：持续接收指令并执行，直到用户退出。

    同一会话上下文会被保留——agent 记得本轮之前说过、做过什么，像聊天一样连续演进。
    """
    print("=== 交互模式（Ctrl+C / Ctrl+D / 输入 exit 退出）===")
    print("agent 会记住本次会话的上下文，可直接接着上一轮继续说。\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[退出] 已退出交互模式。")
            return 0

        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            print("[退出] 已退出交互模式。")
            return 0

        try:
            answer = agent.continue_run(text)
        except AgentError as e:
            print(f"[错误] {e}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n[中断] 已手动停止本次任务，可继续输入。", file=sys.stderr)
            continue

        print(f"\n--- 回答 ---\n{answer}\n")
        print(f"(累计 token: {agent.total_tokens})")


def main() -> None:
    parser = argparse.ArgumentParser(description="本地编码 agent")
    parser.add_argument("task", nargs="?", help="要交给 agent 的任务；省略则进入交互模式")
    parser.add_argument("--steps", type=int, default=30, help="单次任务最大步数（默认 30）")
    parser.add_argument("--cwd", default=".", help="agent 的工作目录（默认当前目录）")
    args = parser.parse_args()

    cwd = Path(args.cwd).expanduser().resolve()
    registry = build_registry(cwd)
    agent = Agent(
        client=LLMClient(Config.from_env()),
        registry=registry,
        system_prompt=SYSTEM_PROMPT,
        max_steps=args.steps,
    )

    # 不带任务 -> 进入交互模式
    if not args.task:
        sys.exit(run_interactive(agent))

    # 单次任务模式
    try:
        answer = agent.run(args.task)
    except AgentError as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 已手动停止 agent。", file=sys.stderr)
        sys.exit(130)

    print("\n========== 最终回答 ==========")
    print(answer)
    print(f"\n(token 总用量: {agent.total_tokens})")


if __name__ == "__main__":
    main()
