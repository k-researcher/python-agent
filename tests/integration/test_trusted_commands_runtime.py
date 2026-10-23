from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.api import supervisor
from agent.config import Settings
from agent.llm import LLMResponse
from agent.tools.advanced import SshExecTool
from agent.tools.base import ToolContext
from agent.tools.builtin import ShellTool
from tests.integration.helpers import authed_client


class ShellLLM:
    """One shell call, then a final answer."""

    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, command: str) -> None:
        self.command = command
        self.calls = 0

    async def chat(self, _m: list[dict[str, Any]], _t: list[dict[str, Any]], **_o: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            call = {
                "id": "call-1",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": self.command})},
            }
            return LLMResponse(None, [call], "tool_calls", 1, 1)
        return LLMResponse("done", [], "stop", 1, 1)


def _wait_for(client: Any, session_id: str, expected: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{session_id}").json()["status"] == expected:
            return
        time.sleep(0.02)
    status = client.get(f"/api/sessions/{session_id}").json()
    raise AssertionError(f"Session did not reach {expected}: {status}")


def _run(tmp_path: Path, command: str, expected: str) -> list[Any]:
    (tmp_path / "README.md").write_text("readme")
    original = supervisor.llm
    supervisor.llm = ShellLLM(command)  # type: ignore[assignment]
    try:
        with authed_client() as client:
            project = client.post(
                "/api/projects", json={"name": "trusted", "root_path": str(tmp_path)}
            ).json()
            session = client.post(
                "/api/sessions",
                json={"project_id": project["id"], "prompt": "list", "mode": "dev"},
            ).json()
            _wait_for(client, session["id"], expected)
            return list(client.get(f"/api/sessions/{session['id']}/approvals").json())
    finally:
        supervisor.llm = original


def test_a_trusted_read_only_command_runs_without_approval(tmp_path: Path) -> None:
    approvals = _run(tmp_path, "ls -la", "completed")
    assert approvals == []


def test_any_other_command_still_waits_for_approval(tmp_path: Path) -> None:
    approvals = _run(tmp_path, "rm README.md", "awaiting_confirmation")
    assert len(approvals) == 1
    assert (tmp_path / "README.md").exists()


@pytest.mark.parametrize("command", ["grep -f.env README.md", "sha256sum --check manifest.txt"])
def test_indirect_reads_wait_for_approval(tmp_path: Path, command: str) -> None:
    approvals = _run(tmp_path, command, "awaiting_confirmation")
    assert len(approvals) == 1


def test_a_link_to_a_secret_waits_for_approval(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("synthetic")
    (tmp_path / "public.txt").symlink_to(tmp_path / ".env")
    approvals = _run(tmp_path, "cat public.txt", "awaiting_confirmation")
    assert len(approvals) == 1


async def test_a_trusted_program_cannot_be_shadowed_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cat").write_text("#!/bin/sh\nprintf shadowed\n")
    (tmp_path / "cat").chmod(0o755)
    (tmp_path / "README.md").write_text("readme")
    monkeypatch.setenv("PATH", str(tmp_path))
    result = await ShellTool().execute(
        ToolContext(session_id="trusted", project_root=tmp_path), {"command": "cat README.md"}
    )
    assert result["stdout"] == "readme"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("df -h", "command -p df -h"),
        ("ls 'two words'", "command -p ls 'two words'"),
        ("cat public.txt", "cat public.txt"),
        ("echo approved", "echo approved"),
    ],
)
async def test_remote_trusted_programs_use_the_system_search_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, expected: str
) -> None:
    connection = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout="ok", stderr=""))
    )
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=connection)
    manager.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("agent.tools.advanced.asyncssh.connect", MagicMock(return_value=manager))
    settings = Settings(
        ssh_connections={"app": {"host": "example.invalid", "known_hosts": "hosts"}}
    )
    result = await SshExecTool(settings).execute(
        ToolContext(session_id="trusted", project_root=tmp_path),
        {"connection": "app", "command": command},
    )
    connection.run.assert_awaited_once_with(expected, check=False, timeout=60)
    assert result["stdout"] == "ok"
