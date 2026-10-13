from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.sandbox import SandboxUnavailableError
from agent.sandbox import prepare as prepare_sandbox
from agent.security import is_sensitive_path
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError

MAX_COMMAND_OUTPUT = 100_000
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9._/~^@{}-]{1,100}$")
SECRET_ERROR = "Access to secret-bearing files is blocked"

# Pathspec excludes for secret-bearing files (first layer, case-insensitive). The output
# filter below is the exact second layer: it applies is_sensitive_path to every file header,
# for example to ".env.local", which a glob cannot separate from ".env.example".
_SENSITIVE_EXCLUDES = tuple(
    f":(exclude,glob,icase){pattern}"
    for pattern in (
        "**/.env",
        "**/.npmrc",
        "**/.pypirc",
        "**/.netrc",
        "**/credentials",
        "**/credentials/**",
        "**/credentials.json",
        "**/id_rsa",
        "**/id_dsa",
        "**/id_ecdsa",
        "**/id_ed25519",
        "**/*.key",
        "**/*.p12",
        "**/*.pfx",
    )
)

_SKIPPED_SENSITIVE = "[skipped: sensitive file]"

# Env vars used to build commands; added so the environment is predictable.
_GIT_ENV = (
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


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Kill the command and everything it spawned (it runs in its own session)."""
    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, 9)
    elif process.returncode is None:
        process.kill()


async def _run_git(context: ToolContext, args: list[str], timeout: int = 60) -> dict[str, Any]:
    """Run git with fixed arguments, no shell, and a sandbox when configured."""
    # Repository configuration can name commands that git runs even for read operations
    # (fsmonitor, hooks, textconv). Turn them off. Clean/smudge filters from the repository
    # configuration cannot be turned off this way: run these tools in the shell sandbox.
    command = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", *args]
    safe_environment: dict[str, str] = {
        name: os.environ[name] for name in _GIT_ENV if name in os.environ
    }
    # Keep git non-interactive and deterministic so the agent never needs a pager.
    # Skip the system configuration: it may be unreadable in a sandbox and can turn
    # on user-level aliases or a pager that change the command output.
    safe_environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
        }
    )

    if context.sandbox is not None:
        try:
            sandboxed = prepare_sandbox(context.sandbox, context.project_root, context.session_id)
        except SandboxUnavailableError as exc:
            raise ToolError(str(exc)) from exc
        if sandboxed is not None:
            prefix, overrides = sandboxed
            command = [*prefix, *command]
            safe_environment.update(overrides)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=context.project_root,
        # No inherited input: the command must not read the terminal of the agent.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=safe_environment,
        start_new_session=os.name == "posix",
    )

    killed_for_output = False

    async def read_stream(stream: asyncio.StreamReader | None) -> tuple[bytes, bool]:
        nonlocal killed_for_output
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
                killed_for_output = True
                _kill_process_tree(process)
                break
        return bytes(collected), truncated

    try:
        stdout_result, stderr_result, _returncode = await asyncio.wait_for(
            asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr),
                process.wait(),
            ),
            timeout=timeout,
        )
    except TimeoutError:
        _kill_process_tree(process)
        await process.wait()
        raise ToolError("Git command timed out") from None
    except asyncio.CancelledError:
        # Stop/shutdown cancels the task; the command must not outlive the session.
        _kill_process_tree(process)
        with suppress(Exception):
            await asyncio.shield(process.wait())
        raise

    stdout, stdout_truncated = stdout_result
    stderr, stderr_truncated = stderr_result
    if process.returncode != 0 and not killed_for_output:
        err_msg = stderr.decode(errors="replace")[:500]
        raise ToolError(f"Git error: {err_msg or 'command failed'}")

    return {
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "truncated": stdout_truncated or stderr_truncated,
    }


def _check_sensitive(value: str) -> None:
    """Raise ToolError when the path names a secret-bearing file."""
    if is_sensitive_path(Path(value)):
        raise ToolError(SECRET_ERROR)


_DIFF_HEADERS = ("diff --git ", "diff --cc ", "diff --combined ")


def _filter_diff_output(output: str) -> str:
    """Replace diff sections of sensitive files with a skip marker.

    A section starts at a "diff --git" header or, for a merge commit, at a combined
    "diff --cc" or "diff --combined" header.
    """
    lines = output.splitlines()
    if not lines:
        return output
    result: list[str] = []
    in_sensitive = False
    for line in lines:
        if line.startswith(_DIFF_HEADERS):
            in_sensitive = _diff_header_is_sensitive(line)
            if in_sensitive:
                result.append(_SKIPPED_SENSITIVE)
                continue
        if not in_sensitive:
            result.append(line)
    return "\n".join(result) + ("\n" if output.endswith("\n") else "")


def _diff_header_is_sensitive(header: str) -> bool:
    """Tell whether a diff header names a sensitive file.

    A header that cannot be parsed counts as sensitive: the section is hidden, not shown.
    """
    for prefix in ("diff --cc ", "diff --combined "):
        if header.startswith(prefix):
            path = header[len(prefix) :]
            if path.startswith('"'):
                return True  # A C-quoted name: do not guess, hide the section.
            return is_sensitive_path(Path(path))
    rest = header[len("diff --git ") :]
    if '"' in rest:
        return True  # A C-quoted name: do not guess, hide the section.
    match = re.match(r"^a/(.*) b/(.*)$", rest)
    if not match:
        return True
    return is_sensitive_path(Path(match.group(1))) or is_sensitive_path(Path(match.group(2)))


def parse_status(stdout: str) -> dict[str, Any]:
    """Parse `git status --porcelain=v1 --branch` text into files and branch."""
    lines = stdout.splitlines()
    if not lines:
        return {"branch": None, "files": []}

    branch: str | None = None
    first = lines[0]
    if first.startswith("##"):
        first = first[3:] if first.startswith("## ") else first[2:]
        if first.startswith("HEAD (no branch"):
            branch = None
        elif first.startswith("No commits yet on "):
            branch = first.split("No commits yet on ", 1)[1]
        else:
            branch = first.split("...", 1)[0].split(" ", 1)[0]

    files: list[dict[str, str | bool]] = []
    for line in lines[1:]:
        if not line or line.startswith("##"):
            continue
        code = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(
            {
                "path": path,
                "index": code[0],
                "worktree": code[1],
                "sensitive": is_sensitive_path(Path(path)),
            }
        )
    return {"branch": branch, "files": files}


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the working tree status."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev", "ask"})
    input_schema = {"type": "object", "additionalProperties": False}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        result = await _run_git(context, ["status", "--porcelain=v1", "--branch"])
        return parse_status(result["stdout"])


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show changes between commits, commit and working tree, etc."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev", "ask"})
    input_schema = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean", "default": False},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 50,
            },
            "context_lines": {
                "type": "integer",
                "minimum": 0,
                "maximum": 20,
                "default": 3,
            },
        },
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        staged = bool(arguments.get("staged", False))
        context_lines = int(arguments.get("context_lines", 3))
        context_lines = max(0, min(20, context_lines))
        raw_paths = arguments.get("paths", [])
        paths = [str(value) for value in raw_paths][:50]

        args = ["diff", f"-U{context_lines}", "--no-color", "--no-ext-diff", "--no-textconv"]
        if staged:
            args.append("--cached")
        args.extend(_pathspec(paths))

        result = await _run_git(context, args)
        return {"diff": _filter_diff_output(result["stdout"]), "truncated": result["truncated"]}


def _validate_paths(paths: list[str]) -> list[str]:
    """Validate path arguments that are always passed after `--`."""
    validated: list[str] = []
    for value in paths:
        if not value or not value.strip() or value.startswith("-"):
            raise ToolError(f"Invalid path: {value!r}")
        normalized = value.replace(os.sep, "/")
        # A directory may end with "/"; any other empty segment is invalid.
        parts = normalized.removesuffix("/").split("/")
        if any(part in ("", "..") for part in parts):
            raise ToolError(f"Invalid path: {value!r}")
        _check_sensitive(value)
        validated.append(value)
    return validated


def _pathspec(paths: list[str]) -> list[str]:
    """Return the arguments after `--`: the paths (or the whole tree) and the secret excludes."""
    return ["--", *(_validate_paths(paths) if paths else ["."]), *_SENSITIVE_EXCLUDES]


class GitLogTool(Tool):
    name = "git_log"
    description = "Show commit logs."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev", "ask"})
    input_schema = {
        "type": "object",
        "properties": {
            "max_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 20,
            },
            "path": {"type": "string"},
        },
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        max_count = int(arguments.get("max_count", 20))
        max_count = max(1, min(200, max_count))
        raw_path = arguments.get("path")
        path: str | None = str(raw_path) if raw_path is not None else None

        # %x1f is the Unit Separator, safe as a field delimiter in commit messages.
        fmt = "%H%x1f%an%x1f%aI%x1f%s%x1f"
        args = ["log", f"-n{max_count}", f"--format={fmt}"]

        if path is not None:
            args.append("--")
            args.extend(_validate_paths([path]))

        result = await _run_git(context, args)
        logs: list[dict[str, str]] = []
        for line in result["stdout"].splitlines():
            parts = line.split("\x1f")
            if len(parts) >= 4:
                logs.append(
                    {
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "subject": parts[3],
                    }
                )

        return {"logs": logs, "truncated": result["truncated"]}


class GitShowTool(Tool):
    name = "git_show"
    description = "Show content of a revision."
    risk_level = RiskLevel.read_only
    allowed_modes = frozenset({"dev", "ask"})
    input_schema = {
        "type": "object",
        "properties": {
            "revision": {"type": "string"},
            "path": {"type": "string"},
            "stat_only": {"type": "boolean", "default": False},
        },
        "required": ["revision"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        revision = str(arguments["revision"])
        # Guard against option injection and validate the revision format.
        if not revision or revision.startswith("-") or not REVISION_PATTERN.fullmatch(revision):
            raise ToolError(f"Invalid revision: {revision!r}")

        stat_only = bool(arguments.get("stat_only", False))
        raw_path = arguments.get("path")
        path: str | None = str(raw_path) if raw_path is not None else None

        args = ["show", "--no-color", "--no-ext-diff", "--no-textconv"]
        if stat_only:
            args.append("--stat")
        # Peel to a commit: a blob or tree revision would print raw file content.
        args.append(f"{revision}^{{commit}}")
        args.extend(_pathspec([path] if path is not None else []))

        result = await _run_git(context, args)
        return {"output": _filter_diff_output(result["stdout"]), "truncated": result["truncated"]}
