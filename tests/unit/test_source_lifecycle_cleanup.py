"""Mocked lifecycle regressions; these do not prove real OS confinement."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from agent import source_lifecycle
from agent.config import Settings
from agent.sandbox import SandboxUnavailableError
from agent.tool_sources import ToolSource
from agent.tools.base import ToolError
from agent.tools.registry import ToolRegistry


class FakeClient:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.reaped = False
        self.start = AsyncMock(side_effect=self._start)
        self.list_tools = AsyncMock(side_effect=self._list_tools)
        self.close = AsyncMock(side_effect=self._close)

    async def _start(self) -> None:
        self.events.append(f"start:{self.name}")

    async def _list_tools(self) -> list[object]:
        self.events.append(f"discover:{self.name}")
        return []

    async def _close(self) -> None:
        self.events.append(f"close:{self.name}")
        self.reaped = True


@dataclass
class LifecycleHarness:
    settings: Settings
    project_root: Path
    registry: ToolRegistry
    sources: tuple[ToolSource, ...]
    clients: tuple[FakeClient, ...]
    events: list[str]
    constructor: Mock
    loader: Mock
    may_connect: AsyncMock

    def context(self):
        return source_lifecycle.source_registry(
            self.settings, "session-cleanup", self.project_root, self.registry
        )

    def assert_closed(self) -> None:
        for client in self.clients:
            client.close.assert_awaited_once_with()
            assert client.reaped


@pytest.fixture
def lifecycle_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., LifecycleHarness]:
    def create(source_count: int = 1, mode: str = "off") -> LifecycleHarness:
        settings = Settings(tool_sources_file=tmp_path / "sources.json", shell_sandbox=mode)
        sources = tuple(
            ToolSource(
                source_id=f"source-{index}",
                workspace_id="workspace-cleanup",
                command=(f"private-server-{index}", "--token=command-secret"),
                env_keys=("SOURCE_CLEANUP_SECRET",),
                enabled=True,
            )
            for index in range(source_count)
        )
        events: list[str] = []
        clients = tuple(FakeClient(source.source_id, events) for source in sources)
        constructor = Mock(side_effect=clients)
        loader = Mock(return_value=sources)
        may_connect = AsyncMock(return_value=True)
        monkeypatch.setattr(source_lifecycle, "StdioMcpClient", constructor)
        monkeypatch.setattr(source_lifecycle, "load_tool_sources", loader)
        monkeypatch.setattr(source_lifecycle, "_may_connect", may_connect)
        return LifecycleHarness(
            settings=settings,
            project_root=tmp_path,
            registry=ToolRegistry(settings=settings, tools=()),
            sources=sources,
            clients=clients,
            events=events,
            constructor=constructor,
            loader=loader,
            may_connect=may_connect,
        )

    return create


@pytest.mark.parametrize("failure_stage", ["start", "list_tools"])
async def test_discovery_failure_closes_client_before_raising(
    lifecycle_factory: Callable[..., LifecycleHarness], failure_stage: str
) -> None:
    harness = lifecycle_factory()
    client = harness.clients[0]
    getattr(client, failure_stage).side_effect = RuntimeError("discovery failed")

    with pytest.raises(ToolError, match="^Trusted MCP sources could not be initialized$"):
        async with harness.context():
            pytest.fail("Discovery failure must not yield a registry")

    client.start.assert_awaited_once_with()
    if failure_stage == "start":
        client.list_tools.assert_not_awaited()
    else:
        client.list_tools.assert_awaited_once_with()
    harness.assert_closed()
    harness.loader.assert_called_once_with(harness.settings.tool_sources_file)
    harness.may_connect.assert_awaited_once_with(
        "session-cleanup", harness.project_root, harness.sources[0]
    )


async def test_second_start_failure_closes_both_clients_in_reverse_order(
    lifecycle_factory: Callable[..., LifecycleHarness],
) -> None:
    harness = lifecycle_factory(source_count=2)
    first_client, second_client = harness.clients
    second_client.start.side_effect = RuntimeError("second startup failed")

    with pytest.raises(ToolError, match="^Trusted MCP sources could not be initialized$"):
        async with harness.context():
            pytest.fail("Second startup failure must not yield a registry")

    first_client.start.assert_awaited_once_with()
    first_client.list_tools.assert_awaited_once_with()
    second_client.start.assert_awaited_once_with()
    second_client.list_tools.assert_not_awaited()
    assert harness.constructor.call_count == 2
    assert harness.events == [
        "start:source-0",
        "discover:source-0",
        "close:source-1",
        "close:source-0",
    ]
    harness.assert_closed()


async def test_body_exception_closes_clients_and_preserves_original_exception(
    lifecycle_factory: Callable[..., LifecycleHarness],
) -> None:
    harness = lifecycle_factory(source_count=2)
    original = RuntimeError("original body failure")

    with pytest.raises(RuntimeError) as caught:
        async with harness.context() as registry:
            assert isinstance(registry, ToolRegistry)
            assert all(not client.reaped for client in harness.clients)
            raise original

    assert caught.value is original
    assert harness.events[-2:] == ["close:source-1", "close:source-0"]
    harness.assert_closed()


@pytest.mark.parametrize("repeated_cancellation", [False, True])
async def test_discovery_cancellation_waits_for_client_cleanup(
    lifecycle_factory: Callable[..., LifecycleHarness], repeated_cancellation: bool
) -> None:
    harness = lifecycle_factory()
    client = harness.clients[0]
    discovery_started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()
    close_tasks: list[asyncio.Task] = []

    async def discover() -> list[object]:
        discovery_started.set()
        await asyncio.Event().wait()
        return []

    async def close() -> None:
        closing_task = asyncio.current_task()
        assert closing_task is not None
        close_tasks.append(closing_task)
        close_started.set()
        await release_close.wait()
        client.reaped = True
        close_finished.set()

    async def run() -> None:
        async with harness.context():
            pytest.fail("Cancelled discovery must not yield a registry")

    client.list_tools.side_effect = discover
    client.close.side_effect = close
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=2)
        assert task.cancel()
        await asyncio.wait_for(close_started.wait(), timeout=2)
        assert not task.done()
        assert not close_finished.is_set()
        if repeated_cancellation:
            for expected_cancellations in (2, 3):
                assert task.cancel()
                await asyncio.sleep(0)
                assert task.cancelling() == expected_cancellations
                assert not task.done()
                assert not close_finished.is_set()
                assert not close_tasks[0].done()
                assert close_tasks[0].cancelling() == 0
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        assert task.cancelled()
        assert close_finished.is_set()
        assert len(close_tasks) == 1
        assert close_tasks[0].done()
        assert not close_tasks[0].cancelled()
        assert close_tasks[0].result() is None
        harness.assert_closed()
    finally:
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
        if close_tasks:
            await asyncio.wait_for(asyncio.gather(*close_tasks, return_exceptions=True), timeout=2)


@pytest.mark.parametrize("mode", ["off", "auto", "required"])
async def test_sandbox_unavailable_is_redacted_and_never_falls_back(
    lifecycle_factory: Callable[..., LifecycleHarness],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    harness = lifecycle_factory(mode=mode)
    client = harness.clients[0]
    secret = "environment-secret-value"
    monkeypatch.setenv("SOURCE_CLEANUP_SECRET", secret)
    client.start.side_effect = SandboxUnavailableError(
        f"private-server-0 --token=command-secret {secret}"
    )

    with pytest.raises(ToolError) as caught:
        async with harness.context():
            pytest.fail("Unavailable sandbox must fail closed")

    assert str(caught.value) == "Trusted MCP sources could not be initialized"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    rendered = "".join(traceback.format_exception(caught.value))
    for sensitive in ("private-server-0", "command-secret", secret):
        assert sensitive not in rendered
    harness.constructor.assert_called_once()
    arguments = harness.constructor.call_args
    assert arguments.args == (list(harness.sources[0].command),)
    assert arguments.kwargs["env"]["SOURCE_CLEANUP_SECRET"] == secret
    assert arguments.kwargs["cwd"] == harness.project_root
    assert arguments.kwargs["session_id"] == "session-cleanup"
    assert arguments.kwargs["sandbox"].mode == "required"
    assert harness.settings.shell_sandbox == mode
    client.start.assert_awaited_once_with()
    client.list_tools.assert_not_awaited()
    harness.assert_closed()


async def test_close_failure_is_redacted_and_other_clients_still_close(
    lifecycle_factory: Callable[..., LifecycleHarness],
) -> None:
    harness = lifecycle_factory(source_count=2)
    first_client, second_client = harness.clients
    secret = "close-secret-value"

    async def fail_close() -> None:
        harness.events.append("close:source-1")
        second_client.reaped = True
        raise RuntimeError(f"private-server-1 --token=command-secret {secret}")

    second_client.close.side_effect = fail_close

    with pytest.raises(ToolError) as caught:
        async with harness.context() as registry:
            assert isinstance(registry, ToolRegistry)

    assert str(caught.value) == "Trusted MCP client could not be closed"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    rendered = "".join(traceback.format_exception(caught.value))
    for sensitive in ("private-server-1", "command-secret", secret):
        assert sensitive not in rendered
    assert harness.events[-2:] == ["close:source-1", "close:source-0"]
    harness.assert_closed()
    assert first_client.reaped


async def test_close_failure_after_cancellation_keeps_cancellation(
    lifecycle_factory: Callable[..., LifecycleHarness],
) -> None:
    harness = lifecycle_factory()
    client = harness.clients[0]
    discovery_started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def discover() -> list[object]:
        discovery_started.set()
        await asyncio.Event().wait()
        return []

    async def fail_close() -> None:
        close_started.set()
        await release_close.wait()
        client.reaped = True
        raise RuntimeError("close failed after cancellation")

    async def run() -> None:
        async with harness.context():
            pytest.fail("Cancelled discovery must not yield a registry")

    client.list_tools.side_effect = discover
    client.close.side_effect = fail_close
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=2)
        assert task.cancel()
        await asyncio.wait_for(close_started.wait(), timeout=2)
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        assert task.cancelled()
        harness.assert_closed()
    finally:
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)


async def test_close_failure_keeps_original_body_exception(
    lifecycle_factory: Callable[..., LifecycleHarness],
) -> None:
    harness = lifecycle_factory()
    client = harness.clients[0]
    original = RuntimeError("original body failure")
    client.close.side_effect = RuntimeError("close failed")

    with pytest.raises(RuntimeError) as caught:
        async with harness.context():
            raise original

    assert caught.value is original
    client.close.assert_awaited_once_with()


async def test_close_failure_during_cancelled_close_keeps_cancellation(
    lifecycle_factory: Callable[..., LifecycleHarness],
) -> None:
    harness = lifecycle_factory()
    client = harness.clients[0]
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def fail_close() -> None:
        close_started.set()
        await release_close.wait()
        client.reaped = True
        raise RuntimeError("close failed after cancellation")

    async def run() -> None:
        async with harness.context():
            pass

    client.close.side_effect = fail_close
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(close_started.wait(), timeout=2)
        assert task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        assert task.cancelled()
        harness.assert_closed()
    finally:
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
