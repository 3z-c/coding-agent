"""编辑类工具：edit_file（精确替换）+ delete_file（删除文件）。

- edit_file：只修改文件某一段时使用，避免整文件重写（省 token、不易丢内容）
- 复用 filesystem._resolve_path 做路径安全
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.base import BaseTool, ToolResult
from tools.filesystem import _resolve_path


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "在文件中把 old_string 精确替换为 new_string（只替换第一处匹配）。"
        "修改文件某一段时优先用它，而不是用 write_file 整文件重写。"
        "old_string 必须与文件内容完全一致（含缩进/空白）；找不到或有多处匹配会报错。"
    )

    parameters = BaseTool.build_schema(
        properties={
            "file_path": BaseTool.string_param("要修改的文件路径（绝对路径或相对工作目录）"),
            "old_string": BaseTool.string_param("要被替换的原文（必须与文件内容逐字符一致）"),
            "new_string": BaseTool.string_param("替换成的新文本"),
        },
        required=["file_path", "old_string", "new_string"],
    )

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root

    def execute(self, arguments: dict) -> ToolResult:
        file_path = self.get_string(arguments, "file_path")
        old = self.get_string(arguments, "old_string")
        new = self.get_string(arguments, "new_string")
        if not file_path:
            return ToolResult.error("缺少 file_path 参数")
        if old is None:
            return ToolResult.error("缺少 old_string 参数")
        if new is None:
            return ToolResult.error("缺少 new_string 参数")

        path = _resolve_path(file_path, self.workspace_root)
        if path is None:
            return ToolResult.error(f"路径越界: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ToolResult.error(f"文件不存在: {file_path}")
        except OSError as e:
            return ToolResult.error(f"读取失败: {e}")

        count = content.count(old)
        if count == 0:
            return ToolResult.error(
                "未找到要替换的原文 old_string，请检查是否与文件内容逐字符一致（尤其缩进/空白）。"
                f"可先用 read_file 查看当前内容。"
            )
        if count > 1:
            return ToolResult.error(
                f"old_string 在文件中出现 {count} 次，替换有歧义，请提供更多上下文使其唯一"
            )

        new_content = content.replace(old, new, 1)
        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"写入失败: {e}")
        return ToolResult.success(f"已把 {path.name} 中的一处原文替换为新文本（文件共 {len(new_content)} 字符）")


class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "删除一个文件。用于清理临时验证文件、多余文件等。只删除文件，不删除目录。"

    parameters = BaseTool.build_schema(
        properties={
            "file_path": BaseTool.string_param("要删除的文件路径（绝对路径或相对工作目录）"),
        },
        required=["file_path"],
    )

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root

    def execute(self, arguments: dict) -> ToolResult:
        file_path = self.get_string(arguments, "file_path")
        if not file_path:
            return ToolResult.error("缺少 file_path 参数")

        path = _resolve_path(file_path, self.workspace_root)
        if path is None:
            return ToolResult.error(f"路径越界: {file_path}")

        if path.is_dir():
            return ToolResult.error(f"{path} 是目录，delete_file 只删除文件；目录请用 run_command 处理")

        try:
            path.unlink()
        except FileNotFoundError:
            return ToolResult.error(f"文件不存在: {file_path}")
        except OSError as e:
            return ToolResult.error(f"删除失败: {e}")
        return ToolResult.success(f"已删除 {path}")
