from __future__ import annotations

import asyncio
import os
import shlex
import signal
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex

from agent.security import PathGuard

if TYPE_CHECKING:
    from agent.safe_fs import SafeProjectFS
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError

MAX_TEXT_BYTES = 1_000_000
MAX_COMMAND_OUTPUT = 100_000


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Kill the command and everything it spawned (it runs in its own session)."""
    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()


def _safe_fs(root: Path) -> SafeProjectFS:
    # Imported lazily: agent.safe_fs imports agent.tools.base, which initialises this package.
    from agent.safe_fs import SafeProjectFS

    return SafeProjectFS(root)


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise ToolError(f"File is larger than {MAX_TEXT_BYTES} bytes")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("File is not valid UTF-8 text") from exc


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories inside the selected project."
    risk_level = RiskLevel.read_only
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "default": "."}},
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        path = PathGuard(context.project_root).resolve(str(arguments.get("path", ".")))
        if not path.is_dir():
            raise ToolError("Path is not a directory")
        entries = [
            {"name": item.name, "type": "directory" if item.is_dir() else "file"}
            for item in sorted(
                path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
            )
        ]
        return {"path": str(path.relative_to(context.project_root)), "entries": entries[:1000]}


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read an UTF-8 text file inside the selected project."
    risk_level = RiskLevel.read_only
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 500},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        path = PathGuard(context.project_root).resolve(str(arguments["path"]))
        if not path.is_file():
            raise ToolError("Path is not a file")
        lines = _read_text(path).splitlines()
        start = max(1, int(arguments.get("start_line", 1)))
        count = min(2000, max(1, int(arguments.get("max_lines", 500))))
        selected = lines[start - 1 : start - 1 + count]
        return {
            "path": str(path.relative_to(context.project_root)),
            "start_line": start,
            "end_line": start + len(selected) - 1,
            "total_lines": len(lines),
            "content": "\n".join(selected),
        }


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search UTF-8 project files using a regular expression."
    risk_level = RiskLevel.read_only
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        root = PathGuard(context.project_root).resolve(str(arguments.get("path", ".")))
        return await asyncio.to_thread(
            self._search,
            context.project_root,
            root,
            str(arguments["pattern"]),
            min(200, int(arguments.get("max_results", 50))),
        )

    @staticmethod
    def _search(project_root: Path, root: Path, pattern_value: str, limit: int) -> dict[str, Any]:
        pattern = regex.compile(pattern_value)
        guard = PathGuard(project_root)
        matches: list[dict[str, Any]] = []
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            excluded = any(part in {".git", ".venv", "node_modules"} for part in path.parts)
            if not path.is_file() or excluded:
                continue
            try:
                guard.resolve(str(path))
                if path.stat().st_size > MAX_TEXT_BYTES:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern.search(line, timeout=0.05):
                        matches.append(
                            {
                                "path": str(path.relative_to(project_root)),
                                "line": number,
                                "text": line[:500],
                            }
                        )
                        if len(matches) >= limit:
                            return {"matches": matches, "truncated": True}
            except (OSError, UnicodeDecodeError, TimeoutError, ValueError):
                continue
        return {"matches": matches, "truncated": False}


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite an UTF-8 text file inside the selected project."
    risk_level = RiskLevel.local_write
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        content = str(arguments["content"])
        relative = _safe_fs(context.project_root).write_text(str(arguments["path"]), content)
        return {"path": str(relative), "bytes_written": len(content.encode("utf-8"))}


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace one exact text fragment in a project file."
    risk_level = RiskLevel.local_write
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        relative, replacements = _safe_fs(context.project_root).edit_text(
            str(arguments["path"]),
            str(arguments["old_text"]),
            str(arguments["new_text"]),
            replace_all=bool(arguments.get("replace_all", False)),
        )
        return {"path": str(relative), "replacements": replacements}


class RemoveFileTool(Tool):
    name = "remove_file"
    description = "Delete one file inside the selected project."
    risk_level = RiskLevel.destructive
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _safe_fs(context.project_root).remove_file(str(arguments["path"]))
        return {"removed": str(relative)}


class ShellTool(Tool):
    name = "shell"
    description = "Run one command without shell expansion in the selected project."
    risk_level = RiskLevel.process_execution
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300, "default": 60},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        command = shlex.split(str(arguments["command"]))
        if not command:
            raise ToolError("Command is empty")
        safe_environment = {
            name: os.environ[name]
            for name in (
                "PATH",
                "HOME",
                "USER",
                "LOGNAME",
                "LANG",
                "LC_ALL",
                "TERM",
                "TMPDIR",
                "TMP",
                "TEMP",
                "SYSTEMROOT",
                "COMSPEC",
                "PATHEXT",
            )
            if name in os.environ
        }
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=context.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=safe_environment,
            start_new_session=os.name == "posix",
        )

        async def read_stream(stream: asyncio.StreamReader | None) -> tuple[bytes, bool]:
            if stream is None:
                return b"", False
            collected = bytearray()
            truncated = False
            while chunk := await stream.read(8192):
                remaining = MAX_COMMAND_OUTPUT - len(collected)
                if remaining > 0:
                    collected.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated = True
                    _kill_process_tree(process)
            return bytes(collected), truncated

        try:
            stdout_result, stderr_result, _returncode = await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout),
                    read_stream(process.stderr),
                    process.wait(),
                ),
                timeout=int(arguments.get("timeout_seconds", 60)),
            )
        except TimeoutError:
            _kill_process_tree(process)
            await process.wait()
            raise ToolError("Command timed out") from None
        except asyncio.CancelledError:
            # Stop/shutdown cancels the task; the command must not outlive the session.
            _kill_process_tree(process)
            with suppress(Exception):
                await asyncio.shield(process.wait())
            raise
        stdout, stdout_truncated = stdout_result
        stderr, stderr_truncated = stderr_result
        return {
            "exit_code": process.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "truncated": stdout_truncated or stderr_truncated,
        }


BUILTIN_TOOLS: tuple[Tool, ...] = (
    ListDirTool(),
    ReadFileTool(),
    SearchFilesTool(),
    WriteFileTool(),
    EditFileTool(),
    RemoveFileTool(),
    ShellTool(),
)
