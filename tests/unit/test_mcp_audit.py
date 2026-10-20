"""Exercise MCP audit ordering and cancellation with fakes, not database persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from agent.mcp_audit import _persist, execute_mcp_attempt
from agent.mcp_client import McpError
from agent.tool_sources import ToolSource
from agent.tools.base import ToolContext, ToolError
from agent.tools.mcp_tools import McpBackedTool, McpToolPolicy

SECRET = "private-remote-secret"
ARGUMENT_SECRET = "private-argument-secret"
COMMAND_SECRET = "private-command-secret"
ENV_KEY = "PRIVATE_MCP_ENV"
RESULT_SECRET = "private-result-secret"
TOOL_CALL_ID = "llm-private-call-id-" + "ю" * 1024


@dataclass(frozen=True)
class AuditRecord:
    channel: str
    destination: str
    operation: str
    payload: str
    status: str
    detail: str | None


class FakeAudit:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.records: list[AuditRecord] = []
        self.writers: list[asyncio.Task[Any]] = []
        self.block_status: str | None = None
        self.fail_status: str | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(
        self,
        channel: str,
        destination: str,
        operation: str,
        payload: str,
        status: str,
        detail: str | None,
    ) -> None:
        writer = asyncio.current_task()
        assert writer is not None
        self.writers.append(writer)
        if status == self.block_status:
            self.entered.set()
            await self.release.wait()
        if status == self.fail_status:
            raise RuntimeError(SECRET)
        self.records.append(AuditRecord(channel, destination, operation, payload, status, detail))
        self.trace.append(status)


class FakeClient:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result = {"content": [{"type": "text", "text": RESULT_SECRET}], "isError": False}
        self.error: Exception | None = None
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def list_tools(self) -> list[Any]:
        raise AssertionError("Discovery is not part of an audited attempt")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        self.trace.append("remote")
        self.entered.set()
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


class Harness:
    def __init__(self, project_root: Path) -> None:
        self.trace: list[str] = []
        self.audit = FakeAudit(self.trace)
        self.client = FakeClient(self.trace)
        self.source = ToolSource(
            source_id="trusted",
            workspace_id="workspace",
            command=(COMMAND_SECRET,),
            env_keys=(ENV_KEY,),
            enabled=True,
        )
        self.tool = McpBackedTool(
            self.client,
            McpToolPolicy(server_id=self.source.source_id),
            "remote.query",
            "Fake remote tool",
            {"type": "object"},
            source=self.source,
        )
        self.context = ToolContext("session", project_root, audit_egress=self.audit)
        self.arguments: dict[str, Any] = {"query": ARGUMENT_SECRET}
        self.guard_error: Exception | None = None

    async def guard(self) -> None:
        assert [record.status for record in self.audit.records] == ["started"]
        assert not self.client.calls
        self.trace.append("guard")
        if self.guard_error is not None:
            raise self.guard_error

    async def execute(self) -> dict[str, Any]:
        return await execute_mcp_attempt(
            self.tool,
            self.context,
            self.arguments,
            tool_call_id=TOOL_CALL_ID,
            before_send=self.guard,
        )

    @asynccontextmanager
    async def running(self) -> AsyncIterator[asyncio.Task[dict[str, Any]]]:
        attempt = asyncio.create_task(self.execute())
        try:
            yield attempt
        finally:
            self.audit.release.set()
            self.client.release.set()
            if not attempt.done():
                attempt.cancel()
            await asyncio.wait_for(asyncio.gather(attempt, return_exceptions=True), timeout=2)

    def assert_pair(self, status: str, outcome: str) -> None:
        assert [record.status for record in self.audit.records] == ["started", status]
        details = []
        for record in self.audit.records:
            assert record.channel == "mcp"
            assert record.destination == self.source.source_id
            assert record.operation == self.tool.name
            assert record.operation != "remote.query"
            assert record.detail is not None
            metadata = json.loads(record.detail)
            assert set(metadata) == {"attempt_id", "tool_call_sha256", "source_revision", "outcome"}
            assert len(metadata["attempt_id"]) == 32
            assert int(metadata["attempt_id"], 16) >= 0
            assert (
                metadata["tool_call_sha256"]
                == hashlib.sha256(TOOL_CALL_ID.encode("utf-8")).hexdigest()
            )
            assert metadata["source_revision"] == self.tool.approval_revision()
            for sensitive in (
                SECRET,
                ARGUMENT_SECRET,
                COMMAND_SECRET,
                ENV_KEY,
                RESULT_SECRET,
                TOOL_CALL_ID,
            ):
                assert sensitive not in record.detail
            details.append(metadata)
        assert details[0]["outcome"] == "started"
        assert details[1] == {**details[0], "outcome": outcome}
        assert self.audit.records[0].payload == self.audit.records[1].payload
        assert all(writer.done() for writer in self.audit.writers)


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


async def wait_entered(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=2)


async def cancel_twice_while_pending(attempt: asyncio.Task[Any]) -> None:
    for _ in range(2):
        assert attempt.cancel()
        await asyncio.sleep(0)
        assert not attempt.done()


async def test_started_precedes_guard_and_remote_and_completed_is_linked(harness: Harness) -> None:
    result = await harness.execute()

    assert harness.trace == ["started", "guard", "remote", "completed"]
    assert harness.client.calls == [("remote.query", harness.arguments)]
    assert result == {"content": harness.client.result["content"], "is_error": False}
    harness.assert_pair("completed", "completed")


async def test_semantic_error_preserves_wire_result_and_records_fixed_outcome(
    harness: Harness,
) -> None:
    harness.client.result["isError"] = True

    result = await harness.execute()

    assert result == {
        "content": [{"type": "text", "text": RESULT_SECRET}],
        "is_error": True,
    }
    assert len(harness.client.calls) == 1
    harness.assert_pair("error", "tool_error")


@pytest.mark.parametrize("error", [McpError(-32000, SECRET), OSError(SECRET), RuntimeError(SECRET)])
async def test_transport_exception_is_sanitized(
    harness: Harness, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    harness.client.error = error

    with pytest.raises(ToolError, match="^MCP invocation failed$") as caught:
        await harness.execute()

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert SECRET not in str(caught.value)
    assert SECRET not in caplog.text
    assert len(harness.client.calls) == 1
    harness.assert_pair("error", "invocation_error")


async def test_guard_denied_never_calls_remote(harness: Harness) -> None:
    harness.guard_error = ToolError(SECRET)

    with pytest.raises(ToolError, match="^MCP invocation failed$"):
        await harness.execute()

    assert harness.trace == ["started", "guard", "error"]
    assert not harness.client.calls
    harness.assert_pair("error", "guard_denied")


async def test_started_write_failure_is_generic_and_never_calls_remote(harness: Harness) -> None:
    harness.audit.fail_status = "started"

    with pytest.raises(ToolError, match="^MCP outbound audit could not be recorded$") as caught:
        await harness.execute()

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert not harness.client.calls
    assert not harness.trace
    assert not harness.audit.records
    assert len(harness.audit.writers) == 1
    assert all(writer.done() for writer in harness.audit.writers)


async def test_missing_audit_callback_never_calls_guard_or_remote(harness: Harness) -> None:
    harness.context = replace(harness.context, audit_egress=None)

    with pytest.raises(ToolError, match="^MCP invocation requires outbound audit$"):
        await harness.execute()

    assert not harness.client.calls
    assert not harness.trace
    assert not harness.audit.writers


async def test_canonical_utf8_payload_and_deep_snapshot_survive_audit_await(
    harness: Harness,
) -> None:
    harness.arguments = {"я": "ёж", "nested": {"items": [{"value": "исходное"}]}, "a": 1}
    original = {"я": "ёж", "nested": {"items": [{"value": "исходное"}]}, "a": 1}
    harness.audit.block_status = "started"

    async with harness.running() as attempt:
        await wait_entered(harness.audit.entered)
        harness.arguments["nested"]["items"][0]["value"] = "изменено"
        harness.arguments["nested"]["items"].append({"value": "новое"})
        harness.arguments["я"] = "другое"
        harness.arguments["extra"] = SECRET
        harness.audit.release.set()
        await asyncio.wait_for(asyncio.shield(attempt), timeout=2)

    sent = harness.client.calls[0][1]
    assert sent == original
    assert sent is not harness.arguments
    assert sent["nested"] is not harness.arguments["nested"]
    assert sent["nested"]["items"] is not harness.arguments["nested"]["items"]
    assert sent["nested"]["items"][0] is not harness.arguments["nested"]["items"][0]
    expected = '{"a":1,"nested":{"items":[{"value":"исходное"}]},"я":"ёж"}'
    assert harness.audit.records[0].payload.encode("utf-8") == expected.encode("utf-8")
    harness.assert_pair("completed", "completed")


async def test_repeated_cancel_during_started_waits_then_records_cancelled(
    harness: Harness,
) -> None:
    harness.audit.block_status = "started"

    async with harness.running() as attempt:
        await wait_entered(harness.audit.entered)
        await cancel_twice_while_pending(attempt)
        assert not harness.client.calls
        assert not harness.audit.records
        assert not harness.audit.writers[0].done()
        harness.audit.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(attempt), timeout=2)

    assert not harness.client.calls
    assert harness.trace == ["started", "cancelled"]
    harness.assert_pair("cancelled", "cancelled")


async def test_cancel_during_remote_waits_for_cancelled_terminal_write(harness: Harness) -> None:
    harness.client.block = True
    harness.audit.block_status = "cancelled"

    async with harness.running() as attempt:
        await wait_entered(harness.client.entered)
        attempt.cancel()
        await wait_entered(harness.audit.entered)
        await cancel_twice_while_pending(attempt)
        assert [record.status for record in harness.audit.records] == ["started"]
        assert not harness.audit.writers[-1].done()
        harness.audit.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(attempt), timeout=2)

    assert len(harness.client.calls) == 1
    harness.assert_pair("cancelled", "cancelled")


@pytest.mark.parametrize("semantic_error", [False, True])
async def test_repeated_cancel_during_terminal_persistence_keeps_terminal_outcome(
    harness: Harness, semantic_error: bool
) -> None:
    harness.client.result["isError"] = semantic_error
    status = "error" if semantic_error else "completed"
    harness.audit.block_status = status

    async with harness.running() as attempt:
        await wait_entered(harness.audit.entered)
        await cancel_twice_while_pending(attempt)
        assert len(harness.client.calls) == 1
        assert not harness.audit.writers[-1].done()
        harness.audit.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(attempt), timeout=2)

    assert len(harness.client.calls) == 1
    harness.assert_pair(status, "tool_error" if semantic_error else "completed")


@pytest.mark.parametrize("status", ["started", "completed", "error", "cancelled"])
async def test_repeated_cancel_then_writer_error_preserves_cancellation_without_orphans(
    harness: Harness, status: str, caplog: pytest.LogCaptureFixture
) -> None:
    harness.audit.block_status = status
    harness.audit.fail_status = status
    harness.client.result["isError"] = status == "error"
    harness.client.block = status == "cancelled"

    async with harness.running() as attempt:
        if status == "cancelled":
            await wait_entered(harness.client.entered)
            attempt.cancel()
        await wait_entered(harness.audit.entered)
        await cancel_twice_while_pending(attempt)
        assert not harness.audit.writers[-1].done()
        harness.audit.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(attempt), timeout=2)

    assert len(harness.client.calls) == (0 if status == "started" else 1)
    assert all(writer.done() for writer in harness.audit.writers)
    assert status not in [record.status for record in harness.audit.records]
    assert SECRET not in caplog.text


@pytest.mark.parametrize("semantic_error", [False, True])
async def test_terminal_write_failure_after_remote_is_generic_and_never_retries(
    harness: Harness, semantic_error: bool, caplog: pytest.LogCaptureFixture
) -> None:
    harness.client.result["isError"] = semantic_error
    harness.audit.fail_status = "error" if semantic_error else "completed"

    with pytest.raises(ToolError, match="^MCP outbound audit could not be recorded$") as caught:
        await harness.execute()

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert SECRET not in caplog.text
    assert len(harness.client.calls) == 1
    assert harness.trace == ["started", "guard", "remote"]
    assert len(harness.audit.writers) == 2
    assert all(writer.done() for writer in harness.audit.writers)


@pytest.mark.parametrize("cancelled", [False, True])
async def test_persist_waits_for_writer_and_reports_cancellation(cancelled: bool) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    writers: list[asyncio.Task[Any]] = []

    async def write() -> None:
        writer = asyncio.current_task()
        assert writer is not None
        writers.append(writer)
        entered.set()
        await release.wait()

    persistence = asyncio.create_task(_persist(write()))
    try:
        await wait_entered(entered)
        if cancelled:
            await cancel_twice_while_pending(persistence)
        assert not writers[0].done()
        release.set()
        assert await asyncio.wait_for(asyncio.shield(persistence), timeout=2) is cancelled
        assert writers[0].done()
    finally:
        release.set()
        if not persistence.done():
            persistence.cancel()
        await asyncio.wait_for(asyncio.gather(persistence, return_exceptions=True), timeout=2)
