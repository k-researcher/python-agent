"""Verify required sandbox preparation before MCP subprocess creation."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from agent.mcp_client import StdioMcpClient
from agent.sandbox import SandboxPolicy, SandboxUnavailableError


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_optional_sandbox_modes_are_rejected(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValueError, match="mode must be 'required'"):
        StdioMcpClient(
            ["server"],
            cwd=tmp_path,
            sandbox=SandboxPolicy(mode=mode),
            session_id="session-1",
        )


@pytest.mark.parametrize("session_id", [None, "", "   "])
def test_required_sandbox_needs_session_id(tmp_path: Path, session_id: str | None) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StdioMcpClient(
            ["server"],
            cwd=tmp_path,
            sandbox=SandboxPolicy(mode="required"),
            session_id=session_id,
        )


@pytest.mark.parametrize("failure", [None, SandboxUnavailableError("no backend")])
async def test_failed_preparation_never_spawns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
) -> None:
    prepare = Mock(return_value=None, side_effect=failure)
    spawn = AsyncMock()
    monkeypatch.setattr("agent.mcp_client.prepare_sandbox", prepare)
    monkeypatch.setattr("agent.mcp_client.asyncio.create_subprocess_exec", spawn)
    client = StdioMcpClient(
        ["server"],
        cwd=tmp_path,
        sandbox=SandboxPolicy(mode="required"),
        session_id="session-1",
    )
    with pytest.raises(SandboxUnavailableError):
        await client.start()
    spawn.assert_not_awaited()


@pytest.mark.parametrize("sandboxed", [False, True])
async def test_spawn_uses_prepared_command_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sandboxed: bool,
) -> None:
    environment = {"HOME": "/original", "PATH": "/bin", "KEEP": "kept"}
    original = dict(environment)
    overrides = {
        "HOME": "/private/home",
        "PATH": "/sandbox/bin",
        "TMPDIR": "/sandbox/tmp",
        "XDG_CONFIG_HOME": "/sandbox/config",
    }
    prepare = Mock(return_value=(["sandbox-exec", "-p", "profile"], overrides))
    process = Mock()
    spawn = AsyncMock(return_value=process)
    initialize = AsyncMock()
    monkeypatch.setattr("agent.mcp_client.prepare_sandbox", prepare)
    monkeypatch.setattr("agent.mcp_client.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr(StdioMcpClient, "_initialize", initialize)
    monkeypatch.setattr(StdioMcpClient, "_read_loop", AsyncMock())
    policy = SandboxPolicy(mode="required") if sandboxed else None
    client = StdioMcpClient(
        ["server", "--stdio"],
        cwd=tmp_path,
        env=environment,
        sandbox=policy,
        session_id="session-1" if sandboxed else None,
    )
    try:
        await client.start()
        await client.start()
        expected = ["sandbox-exec", "-p", "profile"] if sandboxed else []
        assert spawn.call_args.args == (*expected, "server", "--stdio")
        assert spawn.call_args.kwargs["cwd"] == str(tmp_path)
        assert spawn.call_args.kwargs["start_new_session"] is True
        assert spawn.call_args.kwargs["env"] == (
            {**original, **overrides} if sandboxed else original
        )
        assert environment == original
        assert spawn.await_count == initialize.await_count == 1
        assert client._process is process
        if sandboxed:
            prepare.assert_called_once_with(policy, tmp_path, "session-1")
        else:
            prepare.assert_not_called()
    finally:
        if client._reader_task is not None:
            client._reader_task.cancel()
            await asyncio.gather(client._reader_task, return_exceptions=True)
