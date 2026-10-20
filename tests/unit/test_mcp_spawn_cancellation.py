"""Reap a real MCP child when repeated cancellation races with spawn."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from agent.mcp_client import StdioMcpClient

_CHILD_SOURCE = "import sys\nfor line in sys.stdin:\n    pass\n"


async def test_repeated_cancellation_while_spawn_handle_is_hidden_reaps_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = asyncio.Event()
    release = asyncio.Event()
    spawn_cancelled = asyncio.Event()
    create_subprocess_exec = asyncio.create_subprocess_exec
    process: asyncio.subprocess.Process | None = None

    async def delayed_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        nonlocal process
        process = await create_subprocess_exec(*args, **kwargs)
        spawned.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            spawn_cancelled.set()
            raise
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    client = StdioMcpClient(
        [str(Path(sys.executable).resolve()), "-u", "-c", _CHILD_SOURCE],
        cwd=tmp_path,
    )
    task = asyncio.create_task(client.start())
    try:
        await asyncio.wait_for(spawned.wait(), timeout=5)
        assert process is not None
        assert process.returncode is None
        assert client._process is None

        for _attempt in range(2):
            assert task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not spawn_cancelled.is_set()
            assert client._process is None
            assert process.returncode is None

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

        assert task.cancelled()
        assert not spawn_cancelled.is_set()
        assert process.returncode is not None
        await asyncio.wait_for(process.wait(), timeout=3)
        assert client._process is None
        assert client._reader_task is None
        assert not client._pending
    finally:
        release.set()
        try:
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10)
        finally:
            if process is not None:
                if process.returncode is None:
                    process.kill()
                await asyncio.wait_for(process.wait(), timeout=3)
            await asyncio.wait_for(client.close(), timeout=10)


async def test_repeated_cancellation_during_close_reaps_child_and_reply_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_entered = asyncio.Event()
    release_close = asyncio.Event()

    async def initialized() -> None:
        pass

    client = StdioMcpClient(
        [str(Path(sys.executable).resolve()), "-u", "-c", _CHILD_SOURCE], cwd=tmp_path
    )
    monkeypatch.setattr(client, "_initialize", initialized)
    await client.start()
    process = client._process
    assert process is not None
    reply = asyncio.create_task(asyncio.Event().wait())
    client._reply_tasks.add(reply)
    original_close = client._close

    async def delayed_close() -> None:
        close_entered.set()
        await release_close.wait()
        await original_close()

    monkeypatch.setattr(client, "_close", delayed_close)
    task = asyncio.create_task(client.close())
    try:
        await asyncio.wait_for(close_entered.wait(), timeout=3)
        for _attempt in range(2):
            assert task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert process.returncode is not None
        assert reply.cancelled()
        assert not client._reply_tasks
        assert client._process is None
        assert client._reader_task is None
    finally:
        release_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()
        reply.cancel()
        await asyncio.gather(reply, return_exceptions=True)
