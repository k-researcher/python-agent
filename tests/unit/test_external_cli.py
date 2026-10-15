"""Unit tests for the external CLI runner.

The tests use a fake CLI. The fake CLI is a Python script. It prints JSONL
events to stdout. It can also print its environment and sleep on request.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from agent.external_cli import (
    CliRunSpec,
    ExternalEvent,
    build_command,
    parse_event,
    run_external,
)


def _write_script(tmp_path: Path, body: str) -> Path:
    """Write an executable fake CLI and return its path."""
    path = tmp_path / "fake_cli.py"
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _event_sink() -> tuple[list[ExternalEvent], Callable[[ExternalEvent], Awaitable[None]]]:
    """Return a list and an async callback that appends to it."""

    async def callback(event: ExternalEvent) -> None:
        events.append(event)

    events: list[ExternalEvent] = []
    return events, callback


def _run_spec(
    project_root: Path,
    *,
    binary: str = "cli",
    prompt: str = "do the task",
    mode: str = "ask",
    model: str | None = None,
    reasoning: str | None = None,
    forbidden_models: frozenset[str] = frozenset(),
    timeout_seconds: float = 10.0,
) -> CliRunSpec:
    """Return a CliRunSpec with sane defaults for tests."""
    return CliRunSpec(
        binary=binary,
        prompt=prompt,
        project_root=project_root,
        mode=mode,
        model=model,
        reasoning=reasoning,
        timeout_seconds=timeout_seconds,
        forbidden_models=forbidden_models,
    )


SCRIPT_SIMPLE = """\
import json
import sys

lines = [
    {"type": "item.completed", "item": {"type": "agent_message", "text": "one"}},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
    {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "two"}},
    {"type": "turn.completed", "usage": {"in": 10, "out": 5}},
]
for line in lines:
    print(json.dumps(line))
print("not json", file=sys.stderr)
sys.exit(0)
"""


def test_build_command_ask_uses_read_only_sandbox(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, mode="ask")

    command = build_command(spec)

    assert command == [
        "cli",
        "exec",
        "--json",
        "-C",
        str(tmp_path),
        "-s",
        "read-only",
        "do the task",
    ]


def test_build_command_dev_uses_workspace_write_sandbox(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, mode="dev")

    command = build_command(spec)

    assert command == [
        "cli",
        "exec",
        "--json",
        "-C",
        str(tmp_path),
        "-s",
        "workspace-write",
        "do the task",
    ]


def test_build_command_adds_model_and_reasoning(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, mode="dev", model="m-1", reasoning="high")

    command = build_command(spec)

    assert "-m" in command
    assert command[command.index("-m") + 1] == "m-1"
    assert "-c" in command
    assert command[command.index("-c") + 1] == "model_reasoning_effort=high"
    assert command[-1] == "do the task"


def test_build_command_prompt_is_last_argument(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, mode="ask", model="m-2", reasoning="low")

    command = build_command(spec)

    assert command[-1] == "do the task"


def test_build_command_rejects_empty_prompt(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, prompt="")

    with pytest.raises(ValueError, match="prompt"):
        build_command(spec)


def test_build_command_rejects_missing_project_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    spec = _run_spec(missing)

    with pytest.raises(ValueError, match="project root"):
        build_command(spec)


def test_build_command_rejects_forbidden_model(tmp_path: Path) -> None:
    spec = _run_spec(tmp_path, model="blocked", forbidden_models=frozenset({"blocked", "other"}))

    with pytest.raises(ValueError, match="forbidden"):
        build_command(spec)


def test_parse_event_message() -> None:
    line = '{"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "message"
    assert event.text == "hi"


def test_parse_event_reasoning() -> None:
    line = '{"type": "item.completed", "item": {"type": "reasoning", "text": "think"}}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "reasoning"
    assert event.text == "think"


def test_parse_event_command() -> None:
    line = '{"type": "item.completed", "item": {"type": "command_execution", "command": "ls -l"}}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "command"
    assert event.text == "ls -l"


def test_parse_event_file_change() -> None:
    line = '{"type": "item.completed", "item": {"type": "file_change"}}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "file_change"


def test_parse_event_usage() -> None:
    line = '{"type": "turn.completed", "usage": {"in": 3, "out": 4}}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "usage"
    assert event.data == {"in": 3, "out": 4}


def test_parse_event_error() -> None:
    line = '{"type": "error", "message": "bad thing"}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "error"
    assert event.text == "bad thing"


def test_parse_event_turn_failed() -> None:
    line = '{"type": "turn.failed", "message": "no result"}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "error"
    assert event.text == "no result"


def test_parse_event_other() -> None:
    line = '{"type": "something.else", "note": "x"}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "other"


def test_parse_event_empty_line() -> None:
    assert parse_event("") is None
    assert parse_event("   ") is None


def test_parse_event_invalid_json() -> None:
    assert parse_event("not json") is None
    assert parse_event("") is None


def test_parse_event_unknown_item_type() -> None:
    line = '{"type": "item.completed", "item": {"type": "unknown_kind"}}'

    event = parse_event(line)

    assert event is not None
    assert event.kind == "other"


async def test_run_external_success_reports_events_in_order(tmp_path: Path) -> None:
    script = _write_script(tmp_path, SCRIPT_SIMPLE)
    spec = _run_spec(tmp_path, binary=str(script))
    events, callback = _event_sink()

    result = await run_external(spec, callback)

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.final_message == "two"
    assert result.usage == {"in": 10, "out": 5}
    assert len(result.stderr_tail) >= 8 and "not json" in result.stderr_tail
    kinds = [event.kind for event in events]
    assert kinds == ["message", "reasoning", "command", "message", "usage"]


async def test_run_external_nonzero_exit_and_stderr_tail(tmp_path: Path) -> None:
    body = 'import sys\nprint("boom", file=sys.stderr)\nsys.exit(7)\n'
    script = _write_script(tmp_path, body)
    spec = _run_spec(tmp_path, binary=str(script))
    events, callback = _event_sink()

    result = await run_external(spec, callback)

    assert result.exit_code == 7
    assert "boom" in result.stderr_tail
    assert result.events_count == 0


async def test_run_external_timeout_kills_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "sleep.pid"
    body = (
        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)\n"
    )
    script = _write_script(tmp_path, body)
    spec = _run_spec(tmp_path, binary=str(script), timeout_seconds=0.3)
    events, callback = _event_sink()

    result = await run_external(spec, callback)

    assert result.timed_out is True
    assert result.exit_code is not None or result.exit_code is None
    wait_until_dead(pid_file)


async def test_run_external_truncates_very_long_line(tmp_path: Path) -> None:
    chunk = "x" * (1024 * 1024 + 100)
    body = f"import sys\nsys.stdout.write('{{' + {str(chunk)!r} + '}}')\nprint()\n"
    script = _write_script(tmp_path, body)
    spec = _run_spec(tmp_path, binary=str(script))
    events, callback = _event_sink()

    await run_external(spec, callback)

    assert any(event.kind == "other" for event in events)
    truncated = [event for event in events if event.kind == "other"]
    assert truncated and all(len(event.text) <= 1024 * 1024 for event in truncated)


async def test_run_external_external_cancellation_kills_process(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "child.pid"
    body = (
        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)\n"
    )
    script = _write_script(tmp_path, body)
    spec = _run_spec(tmp_path, binary=str(script), timeout_seconds=30.0)
    started = asyncio.Event()

    async def callback(event: ExternalEvent) -> None:
        started.set()

    task = asyncio.create_task(run_external(spec, callback))

    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    wait_until_dead(pid_file)


def wait_until_dead(pid_file: Path) -> None:
    """Wait until the recorded child process is gone."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not pid_file.exists():
            return
        try:
            pid = int(pid_file.read_text(encoding="utf-8"))
        except ValueError:
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError("The child process is still alive")


async def test_run_external_clean_environment(tmp_path: Path) -> None:
    body = (
        "import json, os\n"
        "print(json.dumps({'type': 'item.completed', "
        "'item': {'type': 'agent_message', 'text': json.dumps(dict(os.environ))}}))\n"
    )
    script = _write_script(tmp_path, body)
    secret = "MY_SECRET_TOKEN"
    os.environ[secret] = "s3cret"
    try:
        spec = _run_spec(tmp_path, binary=str(script))
        events, callback = _event_sink()

        env_text = ""
        await run_external(spec, callback)
        if events:
            env_text = events[-1].text
    finally:
        del os.environ[secret]

    assert secret not in env_text
    parsed = json.loads(env_text)
    assert "PATH" in parsed
    assert "MY_SECRET_TOKEN" not in parsed
    for name in ("CLOUD_ACCESS_KEY_ID", "PROVIDER_API_KEY", "MODEL_API_TOKEN"):
        assert name not in parsed


async def test_empty_allow_list_does_not_pass_the_full_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MY_SECRET_TOKEN", "s3cret")
    body = (
        "import json, os\n"
        "print(json.dumps({'type': 'item.completed', "
        "'item': {'type': 'agent_message', 'text': json.dumps(dict(os.environ))}}))\n"
    )
    events, callback = _event_sink()
    await run_external(_run_spec(tmp_path, binary=str(_write_script(tmp_path, body))), callback)
    parsed = json.loads(events[-1].text)
    assert "MY_SECRET_TOKEN" not in parsed
    assert "PATH" not in parsed


async def test_timeout_kills_a_command_that_outlives_the_cli(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    body = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        # The leader exits at once; the child keeps stdout open.
    )
    events, callback = _event_sink()
    spec = _run_spec(tmp_path, binary=str(_write_script(tmp_path, body)), timeout_seconds=1.0)
    result = await run_external(spec, callback)
    assert result.timed_out is True
    wait_until_dead(pid_file)


async def test_failed_event_handler_kills_the_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "cli.pid"
    body = (
        "import json, os, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "print(json.dumps({'type': 'item.completed', "
        "'item': {'type': 'agent_message', 'text': 'hello'}}), flush=True)\n"
        "time.sleep(30)\n"
    )

    async def failing(_event: ExternalEvent) -> None:
        raise RuntimeError("handler failed")

    spec = _run_spec(tmp_path, binary=str(_write_script(tmp_path, body)))
    with pytest.raises(RuntimeError, match="handler failed"):
        await run_external(spec, failing)
    wait_until_dead(pid_file)
