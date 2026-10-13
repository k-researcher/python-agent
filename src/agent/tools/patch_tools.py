from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.security import PathSecurityError
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent.safe_fs import SafeProjectFS

_MAX_FILE_BYTES = 1_000_000
_MAX_PATCH_BYTES = 200_000

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_NEWLINE = "\\ No newline at end of file"


@dataclass(frozen=True, slots=True)
class Hunk:
    old_start: int  # 1-based line number in the old file (0 for an empty old file)
    old_lines: list[str]  # context and removed lines, without the leading " " / "-"
    new_lines: list[str]  # context and added lines, without the leading " " / "+"


@dataclass(frozen=True, slots=True)
class FilePatch:
    old_path: str | None  # None for "--- /dev/null" (new file)
    new_path: str | None  # None for "+++ /dev/null" (deleted file)
    hunks: list[Hunk]


def _restrip(path: str) -> str:
    """Strip the ``a/`` and ``b/`` prefixes and trailing timestamp metadata."""
    name = path
    if "\t" in name:
        name = name.split("\t", 1)[0]
    if name.startswith("a/") or name.startswith("b/"):
        name = name[2:]
    return name.strip()


def _parse_paths(line: str, marker: str) -> str | None:
    raw = line[len(marker) :].strip()
    if raw == "/dev/null":
        return None
    return _restrip(raw)


def _parse_hunk(lines: list[str], i: int) -> tuple[Hunk, int]:
    """Parse one hunk beginning at ``lines[i]`` and return it with the next index."""
    match = _HUNK_RE.match(lines[i])
    if match is None:
        raise ToolError("Malformed hunk header")
    old_start = int(match.group(1))
    # An omitted count means one line (unified diff format).
    old_count = int(match.group(2)) if match.group(2) is not None else 1
    new_count = int(match.group(4)) if match.group(4) is not None else 1
    old_lines: list[str] = []
    new_lines: list[str] = []
    last_list: str | None = None
    j = i + 1
    while j < len(lines):
        line = lines[j]
        if line.startswith("@@") or line.startswith("--- "):
            break
        if line.startswith(_NO_NEWLINE):
            if last_list in ("both", "new") and new_lines and new_lines[-1].endswith("\n"):
                new_lines[-1] = new_lines[-1][:-1]
            if last_list in ("both", "old") and old_lines and old_lines[-1].endswith("\n"):
                old_lines[-1] = old_lines[-1][:-1]
        elif line.startswith("+"):
            new_lines.append(line[1:] + "\n")
            last_list = "new"
        elif line.startswith("-"):
            old_lines.append(line[1:] + "\n")
            last_list = "old"
        elif line.startswith(" "):
            content = line[1:]
            old_lines.append(content + "\n")
            new_lines.append(content + "\n")
            last_list = "both"
        else:
            # Blank line or an unrelated header: end of the hunk.
            break
        j += 1
    if (len(old_lines) != old_count) or (len(new_lines) != new_count):
        raise ToolError(
            f"Malformed hunk: expected -{old_count},+{new_count} lines, "
            f"got -{len(old_lines)},+{len(new_lines)}"
        )
    return Hunk(old_start=old_start, old_lines=old_lines, new_lines=new_lines), j


def parse_unified_diff(text: str) -> list[FilePatch]:
    """Parse a unified diff into a list of :class:`FilePatch` objects.

    Raises :class:`ToolError` if the input is empty, contains no file headers,
    or contains a hunk with line counts that do not match its header.
    """
    if not isinstance(text, str):
        raise ToolError("Patch must be text")
    lines = text.split("\n")
    files: list[FilePatch] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            old_path = _parse_paths(line, "--- ")
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise ToolError("Malformed patch: missing +++ header")
            new_path = _parse_paths(lines[i + 1], "+++ ")
            i += 2
            hunks: list[Hunk] = []
            while i < len(lines) and not lines[i].startswith("--- "):
                if lines[i].startswith("@@"):
                    hunk, i = _parse_hunk(lines, i)
                    hunks.append(hunk)
                else:
                    i += 1
            files.append(FilePatch(old_path=old_path, new_path=new_path, hunks=hunks))
        elif line.strip() == "" or _is_ignored_header(line):
            i += 1
        else:
            raise ToolError("Malformed patch")
    if not files:
        raise ToolError("Patch is empty or contains no files")
    return files


def _is_ignored_header(line: str) -> bool:
    return (
        line.startswith("diff --git")
        or line.startswith("index ")
        or line.startswith("new file mode")
        or line.startswith("deleted file mode")
    )


def apply_hunks(original: str, hunks: Iterable[Hunk], *, path: str) -> str:
    """Apply a sequence of hunks to ``original`` and return the patched text.

    Each hunk is applied at its ``old_start`` position adjusted by the line
    shift produced by the previous hunks. The old lines must match exactly or
    a :class:`ToolError` is raised.
    """
    old = original.splitlines(keepends=True)
    shift = 0
    prev_end = -1
    for n, hunk in enumerate(hunks, 1):
        old_count = len(hunk.old_lines)
        if hunk.old_start - 1 < prev_end:
            raise ToolError(f"{path}: hunks overlap")
        new_count = len(hunk.new_lines)
        # Position in the original file, shifted by earlier hunks. A pure
        # insertion hunk (`-U0`, old_count == 0) with old_start == N > 0 lands
        # AFTER line N, i.e. at 0-based index N, not N-1.
        if old_count == 0 and hunk.old_start > 0:
            pos = hunk.old_start + shift
        else:
            pos = 0 if hunk.old_start == 0 else hunk.old_start - 1 + shift
        if pos > len(old):
            raise ToolError(f"{path}: hunk {n} is outside the file")
        if old[pos : pos + old_count] != hunk.old_lines:
            raise ToolError(f"{path}: hunk {n} does not apply")
        old[pos : pos + old_count] = list(hunk.new_lines)
        shift += new_count - old_count
        prev_end = hunk.old_start - 1 + old_count
    return "".join(old)


def _safe_fs(root: Path) -> SafeProjectFS:
    from agent.safe_fs import SafeProjectFS

    return SafeProjectFS(root)


def _validate_path(path: str) -> str:
    if "\x00" in path:
        raise ToolError("NUL is not allowed in file paths")
    raw = Path(path)
    if raw.is_absolute():
        raise ToolError("Paths must be relative to the project")
    if ".." in raw.parts:
        raise ToolError("Parent traversal is not allowed")
    if not raw.parts:
        raise ToolError("Path must name a file")
    return path


def _try_read(fs: SafeProjectFS, path: str) -> str | None:
    try:
        return fs.read_text(path, max_bytes=_MAX_FILE_BYTES)
    except ToolError:
        # Distinguish a missing target (file or any parent directory) from
        # other errors using a safe, symlink-free existence check from the
        # filesystem layer instead of matching on error-message text.
        if not fs.exists(path):
            return None
        raise


class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = "Apply a unified diff to project files atomically."
    risk_level = RiskLevel.local_write
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "patch": {"type": "string", "maxLength": _MAX_PATCH_BYTES},
        },
        "required": ["patch"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        patch = str(arguments["patch"])
        if len(patch) > _MAX_PATCH_BYTES:
            raise ToolError(f"Patch exceeds {_MAX_PATCH_BYTES} characters")
        patches = parse_unified_diff(patch)
        try:
            return await asyncio.to_thread(self._apply, context.project_root, patches)
        except PathSecurityError as exc:
            raise ToolError(str(exc)) from exc

    @staticmethod
    def _apply(root: Path, patches: list[FilePatch]) -> dict[str, Any]:
        fs = _safe_fs(root)
        # Pass one: read everything and apply all hunks in memory. Any failure
        # raises before a single file is touched.
        planned: list[tuple[str, str, str | None, int]] = []
        for patch in patches:
            if patch.old_path is None:
                action = "create"
                new_path = patch.new_path
                assert new_path is not None
                path = _validate_path(new_path)
                if _try_read(fs, path) is not None:
                    raise ToolError(f"{path}: file already exists")
                new_content = apply_hunks("", patch.hunks, path=path)
            elif patch.new_path is None:
                action = "delete"
                old_path = patch.old_path
                assert old_path is not None
                path = _validate_path(old_path)
                existing = _try_read(fs, path)
                if existing is None:
                    raise ToolError(f"{path}: file does not exist")
                content = apply_hunks(existing, patch.hunks, path=path)
                if content != "":
                    raise ToolError(f"{path}: deletion did not empty the file")
                new_content = None
            else:
                action = "modify"
                new_path = patch.new_path
                assert new_path is not None
                path = _validate_path(new_path)
                existing = _try_read(fs, path)
                if existing is None:
                    raise ToolError(f"{path}: file does not exist")
                new_content = apply_hunks(existing, patch.hunks, path=path)
            planned.append((path, action, new_content, len(patch.hunks)))
        # Pass two: write everything out.
        for path, action, new_content, _count in planned:
            if action == "delete":
                fs.remove_file(path)
            else:
                assert new_content is not None
                fs.write_text(path, new_content)
        return {
            "files": [{"path": item[0], "action": item[1], "hunks": item[3]} for item in planned]
        }


class EditByLinesTool(Tool):
    name = "edit_by_lines"
    description = (
        "Replace or insert lines in a project file. end_line == start_line - 1 "
        "inserts new_text before start_line."
    )
    risk_level = RiskLevel.local_write
    allowed_modes = frozenset({"dev"})
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 0},
            "new_text": {"type": "string"},
            "expected_old_text": {"type": "string"},
        },
        "required": ["path", "start_line", "end_line", "new_text"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        expected = arguments.get("expected_old_text")
        try:
            return await asyncio.to_thread(
                self._edit,
                context.project_root,
                str(arguments["path"]),
                int(arguments["start_line"]),
                int(arguments["end_line"]),
                str(arguments["new_text"]),
                None if expected is None else str(expected),
            )
        except PathSecurityError as exc:
            raise ToolError(str(exc)) from exc

    @staticmethod
    def _edit(
        root: Path,
        path_value: str,
        start_line: int,
        end_line: int,
        new_text: str,
        expected_old_text: str | None,
    ) -> dict[str, Any]:
        fs = _safe_fs(root)
        original = fs.read_text(path_value, max_bytes=_MAX_FILE_BYTES)
        if start_line < 1:
            raise ToolError("start_line must be >= 1")
        if end_line < start_line - 1:
            raise ToolError("end_line must be >= start_line - 1")
        lines = original.splitlines(keepends=True)
        total = len(lines)
        if end_line == start_line - 1:
            # Insertion before start_line; the insertion point may sit just past
            # the last line.
            if start_line > total + 1:
                raise ToolError("Insertion point is outside the file")
            start_idx = start_line - 1
            end_idx = start_line - 1
            # Insertion after the last line of a file without a trailing
            # newline must separate the new text onto its own line.
            if start_line == total + 1 and total > 0 and not lines[-1].endswith("\n"):
                new_text = "\n" + new_text
        else:
            if end_line > total:
                raise ToolError("end_line is outside the file")
            start_idx = start_line - 1
            end_idx = end_line
        replaced_text = "".join(lines[start_idx:end_idx])
        if expected_old_text is not None and replaced_text != expected_old_text:
            raise ToolError("expected_old_text does not match the current lines")
        new_content = "".join(lines[:start_idx]) + new_text + "".join(lines[end_idx:])
        # Single atomic pass: replace_if_unchanged re-reads the file and
        # verifies its content still matches `original`, closing the TOCTOU
        # window between this first read and the write (and supporting empty
        # files, which edit_text rejects).
        relative = fs.replace_if_unchanged(path_value, original, new_content)
        return {
            "path": str(relative),
            "replaced_lines": end_idx - start_idx,
            "inserted_lines": len(new_text.splitlines()),
        }


__all__ = [
    "ApplyPatchTool",
    "EditByLinesTool",
    "FilePatch",
    "Hunk",
    "apply_hunks",
    "parse_unified_diff",
]
