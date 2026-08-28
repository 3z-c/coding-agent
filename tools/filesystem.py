"""文件系统工具：read_file / write_file / list_dir。

路径安全限制：如果指定了 workspace_root，所有路径都必须位于该目录内，
防止模型乱读系统文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.base import BaseTool, ToolResult


def _resolve_path(path_str: str, root: Optional[Path]) -> Optional[Path]:
    """把模型给的路径解析为绝对路径；若指定 root，则限制在 root 内。

    规则：
    - 空路径 -> None
    - 相对路径相对于 root（或当前目录）解析
    - 绝对路径直接使用，但若 root 存在则必须位于 root 内，否则返回 None（越界）
    """
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        if root is None:
            p = Path.cwd() / p
        else:
            p = root / p
    p = p.resolve()
    if root is not None:
        try:
            p.relative_to(root.resolve())
        except ValueError:
            return None  # 越界
    return p


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取指定文本文件的内容。用于查看源代码、配置、日志等。返回行数受 max_lines 限制。"

    parameters = BaseTool.build_schema(
        properties={
            "file_path": BaseTool.string_param("要读取的文件路径（绝对路径或相对于工作目录）"),
            "max_lines": BaseTool.int_param("最多返回的行数，超出部分截断（默认 200）"),
        },
        required=["file_path"],
    )

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root

    def execute(self, arguments: dict) -> ToolResult:
        file_path = self.get_string(arguments, "file_path")
        max_lines = self.get_int(arguments, "max_lines", 200)
        if not file_path:
            return ToolResult.error("缺少 file_path 参数")
        path = _resolve_path(file_path, self.workspace_root)
        if path is None:
            return ToolResult.error(f"路径越界或不存在: {file_path}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ToolResult.error(f"文件不存在: {file_path}")
        except IsADirectoryError:
            return ToolResult.error(f"{file_path} 是一个目录，请用 list_dir 查看目录内容")
        except OSError as e:
            return ToolResult.error(f"读取失败: {e}")

        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines]) + f"\n...[已截断，共 {len(lines)} 行，仅显示前 {max_lines} 行]"
        return ToolResult.success(content)


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "写入（或覆盖）一个文本文件。父目录不存在时会自动创建。"

    parameters = BaseTool.build_schema(
        properties={
            "file_path": BaseTool.string_param("要写入的文件路径（绝对路径或相对于工作目录）"),
            "content": BaseTool.string_param("要写入的文件内容"),
        },
        required=["file_path", "content"],
    )

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root

    def execute(self, arguments: dict) -> ToolResult:
        file_path = self.get_string(arguments, "file_path")
        content = self.get_string(arguments, "content")
        if not file_path:
            return ToolResult.error("缺少 file_path 参数")
        if content is None:
            return ToolResult.error("缺少 content 参数")
        path = _resolve_path(file_path, self.workspace_root)
        if path is None:
            return ToolResult.error(f"路径越界: {file_path}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"写入失败: {e}")
        return ToolResult.success(f"已写入 {len(content)} 个字符到 {path}")


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "列出目录下的条目（文件和子目录名），用于浏览目录结构。"

    parameters = BaseTool.build_schema(
        properties={
            "path": BaseTool.string_param("要列出的目录路径，省略则列出工作目录", ),
            "max_entries": BaseTool.int_param("最多列出多少个条目（默认 100）"),
        },
    )

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root

    def execute(self, arguments: dict) -> ToolResult:
        path_str = self.get_string(arguments, "path")
        max_entries = self.get_int(arguments, "max_entries", 100)
        path = _resolve_path(path_str if path_str else ".", self.workspace_root)
        if path is None:
            return ToolResult.error(f"路径越界: {path_str}")
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except FileNotFoundError:
            return ToolResult.error(f"目录不存在: {path}")
        except NotADirectoryError:
            return ToolResult.error(f"{path} 不是目录")
        except OSError as e:
            return ToolResult.error(f"列出失败: {e}")

        lines = []
        for p in entries[:max_entries]:
            kind = "dir " if p.is_dir() else "file"
            lines.append(f"{kind}  {p.name}")
        if len(entries) > max_entries:
            lines.append(f"...[共 {len(entries)} 个条目，仅显示前 {max_entries} 个]")
        return ToolResult.success("\n".join(lines) if lines else "(空目录)")
