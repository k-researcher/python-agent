"""Runs an external CLI as a child process and reads its JSONL events.

An external CLI reads a task and prints events to stdout. Each event is one
JSON object per line. This module starts the process, parses each line and
forwards the events to a callback. It returns a summary of the run.

This module is decoupled from the agent runtime. It needs no API access.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent.modes import tool_profile

EventKind = Literal["message", "reasoning", "command", "file_change", "usage", "error", "other"]

_CMD = "exec"
_JSON_FLAG = "--json"
_CHANGE_FLAG = "-C"
_SANDBOX_FLAG = "-s"
_MODEL_FLAG = "-m"
_CONFIG_FLAG = "-c"
_REASONING_CONFIG = "model_reasoning_effort={level}"
_SANDBOX_ASK = "read-only"
_SANDBOX_DEV = "workspace-write"

_MAX_LINE_LENGTH = 1024 * 1024
_STDERR_TAIL_LIMIT = 2000
_KILL_GRACE_SECONDS = 2.0

# The child process gets only these environment variables. They are enough to
# run a program. Credential files and secret variables stay private.
_ALLOWED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


@dataclass(frozen=True, slots=True)
class CliRunSpec:
    """The settings for one external CLI run."""

    binary: str
    prompt: str
    project_root: Path
    mode: Literal["ask", "dev"]
    model: str | None = None
    reasoning: str | None = None
    timeout_seconds: float = 60.0
    forbidden_models: frozenset[str] = frozenset()


def build_command(spec: CliRunSpec) -> list[str]:
    """Return the command line for the external CLI.

    The sandbox is read-only for ask mode.
    The sandbox is workspace-write for dev mode.
    The prompt is the last argument.
    """
    if not spec.prompt:
        raise ValueError("The prompt must not be empty")
    if not spec.project_root.exists():
        raise ValueError("The project root must exist")
    if spec.model in spec.forbidden_models:
        raise ValueError("This model is forbidden")
    sandbox = _SANDBOX_ASK if tool_profile(spec.mode) == "ask" else _SANDBOX_DEV
    command = [
        spec.binary,
        _CMD,
        _JSON_FLAG,
        _CHANGE_FLAG,
        str(spec.project_root),
        _SANDBOX_FLAG,
        sandbox,
    ]
    if spec.model is not None:
        command.extend([_MODEL_FLAG, spec.model])
    if spec.reasoning is not None:
        command.extend([_CONFIG_FLAG, _REASONING_CONFIG.format(level=spec.reasoning)])
    command.append(spec.prompt)
    return command


@dataclass(frozen=True, slots=True)
class ExternalEvent:
    """One event that the external CLI reported."""

    kind: EventKind
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def parse_event(line: str) -> ExternalEvent | None:
    """Parse one JSONL line into an event.

    Return None for an empty line or a line that is not JSON.
    Map the known item and turn types to a single kind.
    """
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    item = payload.get("item")
    item_type = item.get("type") if isinstance(item, dict) else None
    if payload.get("type") == "item.completed" and isinstance(item_type, str):
        return _parse_item_completed(payload, item_type)
    if payload.get("type") == "turn.completed":
        raw_usage = payload.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        return ExternalEvent(kind="usage", data=dict(usage))
    message = _error_message(payload)
    if message is not None:
        return ExternalEvent(kind="error", text=message)
    return ExternalEvent(kind="other", data=payload)


def _parse_item_completed(payload: dict[str, Any], item_type: str) -> ExternalEvent:
    """Map an item.completed payload to an event."""
    item = payload.get("item")
    if not isinstance(item, dict):
        return ExternalEvent(kind="other", data=payload)
    if item_type == "agent_message":
        raw_text = item.get("text")
        text = raw_text if isinstance(raw_text, str) else ""
        return ExternalEvent(kind="message", text=text)
    if item_type == "reasoning":
        raw_text = item.get("text")
        text = raw_text if isinstance(raw_text, str) else ""
        return ExternalEvent(kind="reasoning", text=text)
    if item_type == "command_execution":
        command = item.get("command")
        text = command if isinstance(command, str) else ""
        return ExternalEvent(kind="command", text=text)
    if item_type == "file_change":
        return ExternalEvent(kind="file_change")
    return ExternalEvent(kind="other", data=payload)


def _error_message(payload: dict[str, Any]) -> str | None:
    """Return a text for an error event, or None for another event type."""
    if payload.get("type") == "error":
        message = payload.get("message")
        if isinstance(message, str):
            return message
        return ""
    if payload.get("type") == "turn.failed":
        message = payload.get("message")
        if isinstance(message, str):
            return message
        return ""
    return None


@dataclass(frozen=True, slots=True)
class ExternalResult:
    """The summary of one external CLI run."""

    exit_code: int | None
    final_message: str
    events_count: int
    usage: dict[str, Any]
    timed_out: bool
    cancelled: bool
    stderr_tail: str


async def run_external(
    spec: CliRunSpec,
    on_event: Callable[[ExternalEvent], Awaitable[None]],
) -> ExternalResult:
    """Run the external CLI and forward its events.

    The process runs with a clean environment and no standard input.
    A line longer than 1 MiB is truncated and reported as an other event.
    A timeout kills the whole process group. An external cancellation kills
    the group as well and then propagates.
    """
    env = _clean_env()
    command = build_command(spec)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=spec.project_root,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        env=env,
    )
    storage = _RunStorage()
    reader_task = asyncio.create_task(_consume_stdout(process.stdout, on_event, storage))
    stderr_task = asyncio.create_task(_consume_stderr(process.stderr, storage))
    try:
        async with asyncio.timeout(spec.timeout_seconds):
            # Wait for all three at once: a failed reader (for example a failed event
            # handler) must stop the run at once, not after the process ends.
            exit_code, _, _ = await asyncio.gather(process.wait(), reader_task, stderr_task)
        return storage.result(exit_code, timed_out=False, cancelled=False)
    except TimeoutError:
        await _terminate_group(process)
        return storage.result(exit_code=process.returncode, timed_out=True, cancelled=False)
    except BaseException:
        # Cancellation or a failed event handler: never leave the process group running.
        await _terminate_group(process)
        raise
    finally:
        for task in (reader_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(reader_task, stderr_task, return_exceptions=True)


def _clean_env() -> dict[str, str]:
    """Return the child environment with a small allow-list.

    The allow-list keeps the process runnable and the environment private.
    Secret variables and credential files never reach the child process.
    The result is a dict even when it is empty: None would pass the full environment.
    """
    return {name: os.environ[name] for name in _ALLOWED_ENV if name in os.environ}


class _RunStorage:
    """Collect events and stderr text during one run."""

    def __init__(self) -> None:
        self.events: list[ExternalEvent] = []
        self.stderr_parts: list[bytes] = []
        self.stderr_size = 0
        self.usage: dict[str, Any] = {}

    def record(self, event: ExternalEvent) -> None:
        self.events.append(event)
        if event.kind == "usage":
            self.usage = event.data

    def _final_message(self) -> str:
        message_events = [event for event in self.events if event.kind == "message"]
        if not message_events:
            return ""
        return message_events[-1].text

    def _stderr_tail(self) -> str:
        collected = b"".join(self.stderr_parts)
        return collected[-_STDERR_TAIL_LIMIT:].decode("utf-8", errors="replace")

    def result(
        self,
        exit_code: int | None,
        *,
        timed_out: bool,
        cancelled: bool,
    ) -> ExternalResult:
        """Build the run summary from the collected state."""
        return ExternalResult(
            exit_code=exit_code,
            final_message=self._final_message(),
            events_count=len(self.events),
            usage=dict(self.usage),
            timed_out=timed_out,
            cancelled=cancelled,
            stderr_tail=self._stderr_tail(),
        )

    def append_stderr(self, chunk: bytes) -> None:
        self.stderr_parts.append(chunk)
        self.stderr_size += len(chunk)


async def _consume_stdout(
    stream: asyncio.StreamReader | None,
    on_event: Callable[[ExternalEvent], Awaitable[None]],
    storage: _RunStorage,
) -> None:
    """Read stdout by line and forward each parsed event."""
    if stream is None:
        return
    buffer = bytearray()
    truncated_pending = False
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            if truncated_pending:
                await _emit_truncated(storage, on_event)
            elif buffer:
                await _handle_line(storage, on_event, bytes(buffer))
            break
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if truncated_pending:
                truncated_pending = False
                await _emit_truncated(storage, on_event)
            else:
                await _handle_line(storage, on_event, raw)
        if truncated_pending:
            # The oversize line has no end yet. Drop its partial data.
            buffer.clear()
        elif len(buffer) > _MAX_LINE_LENGTH:
            truncated_pending = True
            buffer.clear()


async def _handle_line(
    storage: _RunStorage,
    on_event: Callable[[ExternalEvent], Awaitable[None]],
    raw: bytes,
) -> None:
    """Parse one line and forward it as an event.

    The line is the bytes up to one newline. It may be huge.
    A huge line is truncated and reported as an other event.
    """
    text = raw.decode("utf-8", errors="replace").rstrip("\r")
    if len(text) > _MAX_LINE_LENGTH:
        await _emit_truncated(storage, on_event)
        return
    event = parse_event(text)
    if event is not None:
        storage.record(event)
        await on_event(event)


async def _emit_truncated(
    storage: _RunStorage,
    on_event: Callable[[ExternalEvent], Awaitable[None]],
) -> None:
    """Report that one line was too long and was dropped."""
    event = ExternalEvent(kind="other", text="Line was truncated")
    storage.record(event)
    await on_event(event)


async def _consume_stderr(stream: asyncio.StreamReader | None, storage: _RunStorage) -> None:
    """Read stderr and keep only its tail."""
    if stream is None:
        return
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        storage.append_stderr(chunk)
        if storage.stderr_size > _STDERR_TAIL_LIMIT * 4:
            keep = b"".join(storage.stderr_parts)[-_STDERR_TAIL_LIMIT:]
            storage.stderr_parts = [keep]
            storage.stderr_size = len(keep)


async def _terminate_group(process: asyncio.subprocess.Process) -> None:
    """Stop the whole process group: SIGTERM, a grace period, then SIGKILL.

    The group can outlive its leader, for example a command that the CLI started.
    Thus the group gets the signals also when the leader has already exited.
    """
    group_id = process.pid
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group_id, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_SECONDS)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group_id, signal.SIGKILL)
    # Reap the leader so that no zombie process stays.
    await process.wait()
