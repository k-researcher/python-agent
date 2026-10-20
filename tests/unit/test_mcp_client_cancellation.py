"""Exercise real task cancellation with a live MCP child process."""

import asyncio
import sys
from pathlib import Path

import pytest

from agent.mcp_client import StdioMcpClient

_CHILD_SOURCE = "import sys\nfor line in sys.stdin:\n    pass\n"


@pytest.mark.parametrize("during_initialize", [True, False])
async def test_cancellation_reaps_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    during_initialize: bool,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def initialize(client: StdioMcpClient) -> None:
        if during_initialize:
            entered.set()
            await release.wait()

    monkeypatch.setattr(StdioMcpClient, "_initialize", initialize)
    client = StdioMcpClient([sys.executable, "-u", "-c", _CHILD_SOURCE], cwd=tmp_path)

    async def run() -> None:
        async with client:
            entered.set()
            await release.wait()

    task = asyncio.create_task(run())
    process: asyncio.subprocess.Process | None = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        process = client._process
        assert process is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(process.wait(), timeout=3)
        assert process.returncode is not None
        assert client._process is None
        assert client._reader_task is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await client.close()
