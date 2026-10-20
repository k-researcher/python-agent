"""Verify MCP audit persistence and the final pre-egress guard with real SQL."""

import asyncio
import hashlib
import json
from typing import Any

import pytest
from sqlalchemy import select

from agent.database import session_factory
from agent.mcp_client import McpError
from agent.models import Message, OutboundAudit, Session, ToolCall, User
from agent.session_fence import CURRENT_FENCE, FenceLostError, claim
from tests.integration.test_runtime_source_revision import (
    RuntimeCase,
    add_call,
    load_session,
)
from tests.integration.test_runtime_source_revision import runtime_case as runtime_case


async def audit_rows(case: RuntimeCase) -> list[OutboundAudit]:
    async with session_factory() as db:
        return list(
            await db.scalars(
                select(OutboundAudit)
                .where(OutboundAudit.session_id == case.session_id, OutboundAudit.category == "mcp")
                .order_by(OutboundAudit.id)
            )
        )


async def execute(case: RuntimeCase, call_id: str) -> bool:
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        call = await db.get(ToolCall, call_id)
        assert call is not None
        return await case.supervisor._execute_tool_call(db, session, call)


async def test_mcp_call_has_durable_started_and_linked_terminal_without_raw_payload(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = runtime_case
    arguments = {"query": "секрет-☃", "nested": {"limit": 2}}
    call_id = await add_call(case, revision=case.tool.approval_revision())
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None
        call.arguments = arguments
        await db.commit()

    async def remote(name: str, sent: dict[str, Any]) -> dict[str, Any]:
        case.client.calls += 1
        rows = await audit_rows(case)
        assert len(rows) == 1 and rows[0].status == "started"
        assert sent == arguments
        return {"content": [{"type": "text", "text": "remote-result-canary"}]}

    monkeypatch.setattr(case.client, "call_tool", remote)
    assert await execute(case, call_id)
    rows = await audit_rows(case)
    assert [row.status for row in rows] == ["started", "completed"]
    encoded = json.dumps(
        arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    details = [json.loads(row.detail) for row in rows]
    assert details[0]["attempt_id"] == details[1]["attempt_id"]
    assert details[0]["tool_call_sha256"] == hashlib.sha256(call_id.encode()).hexdigest()
    assert details[0]["source_revision"] == case.tool.approval_revision()
    for row, detail in zip(rows, details, strict=True):
        assert row.destination == case.source.source_id
        assert row.operation == case.tool.name
        assert row.payload_bytes == len(encoded)
        assert row.payload_sha256 == hashlib.sha256(encoded).hexdigest()
        assert set(detail) == {"attempt_id", "tool_call_sha256", "source_revision", "outcome"}
        assert "секрет" not in row.detail
        assert "remote-result-canary" not in row.detail
        assert call_id not in row.detail
    assert case.client.calls == 1


@pytest.mark.parametrize("change", ["membership", "stop", "schema"])
async def test_recheck_after_started_audit_prevents_revoked_egress(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    original = case.supervisor._audit_egress

    async def audit(*args: Any) -> None:
        await original(*args)
        if args[5] != "started":
            return
        if change == "schema":
            case.tool.input_schema["properties"] = {"new": {"type": "string"}}
        else:
            async with session_factory() as db:
                if change == "membership":
                    user = await db.get(User, case.user_id)
                    assert user is not None
                    user.enabled = False
                else:
                    session = await db.get(Session, case.session_id)
                    assert session is not None
                    session.stop_requested = True
                await db.commit()

    monkeypatch.setattr(case.supervisor, "_audit_egress", audit)
    assert await execute(case, call_id)
    assert case.client.calls == 0
    rows = await audit_rows(case)
    assert [row.status for row in rows] == ["started", "error"]
    assert json.loads(rows[-1].detail)["outcome"] == "guard_denied"


@pytest.mark.parametrize("stage", ["started", "completed"])
async def test_audit_failure_never_retries_or_leaks_error(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    original = case.supervisor._audit_egress

    async def audit(*args: Any) -> None:
        if args[5] == stage:
            raise RuntimeError("database-password-canary")
        await original(*args)

    monkeypatch.setattr(case.supervisor, "_audit_egress", audit)
    assert await execute(case, call_id)
    assert not await execute(case, call_id)
    assert case.client.calls == (1 if stage == "completed" else 0)
    rows = await audit_rows(case)
    assert [row.status for row in rows] == (["started"] if stage == "completed" else [])
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.status == "error"
        assert "database-password-canary" not in json.dumps(call.result)


async def test_remote_exception_is_not_saved_in_audit_or_runtime_error(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())

    async def remote(*args: Any) -> dict[str, Any]:
        case.client.calls += 1
        raise McpError(-1, "credential-canary")

    monkeypatch.setattr(case.client, "call_tool", remote)
    assert await execute(case, call_id)
    rows = await audit_rows(case)
    assert [row.status for row in rows] == ["started", "error"]
    assert json.loads(rows[-1].detail)["outcome"] == "invocation_error"
    assert all("credential-canary" not in row.detail for row in rows)
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        messages = list(await db.scalars(select(Message).where(Message.tool_call_id == call_id)))
        assert call is not None and call.status == "error"
        assert "credential-canary" not in json.dumps(call.result)
        assert all("credential-canary" not in (message.content or "") for message in messages)


async def steal_owner(case: RuntimeCase) -> None:
    token = CURRENT_FENCE.set(None)
    try:
        async with session_factory() as db:
            session = await db.get(Session, case.session_id)
            assert session is not None
            session.owner_token = "another-owner"
            await db.commit()
    finally:
        CURRENT_FENCE.reset(token)


@pytest.mark.parametrize("stage", ["started", "remote"])
async def test_lost_fence_keeps_started_without_stale_writes_or_self_deadlock(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    original_audit = case.supervisor._audit_egress

    async def audit(*args: Any) -> None:
        await original_audit(*args)
        if stage == "started" and args[5] == "started":
            await steal_owner(case)

    async def remote(*args: Any) -> dict[str, Any]:
        case.client.calls += 1
        await steal_owner(case)
        return {"content": []}

    monkeypatch.setattr(case.supervisor, "_audit_egress", audit)
    if stage == "remote":
        monkeypatch.setattr(case.client, "call_tool", remote)
    async with session_factory() as db:
        fence = await claim(db, case.session_id, lease_seconds=60)
    assert fence is not None
    token = CURRENT_FENCE.set(fence)
    try:
        with pytest.raises(FenceLostError):
            await asyncio.wait_for(execute(case, call_id), timeout=3)
    finally:
        CURRENT_FENCE.reset(token)
    assert case.client.calls == (1 if stage == "remote" else 0)
    rows = await audit_rows(case)
    assert [row.status for row in rows] == ["started"]
    async with session_factory() as db:
        session = await db.get(Session, case.session_id)
        call = await db.get(ToolCall, call_id)
        assert session is not None and session.owner_token == "another-owner"
        assert call is not None and call.status == "running"


async def test_cancellation_records_terminal_and_recovery_does_not_repeat_remote(
    runtime_case: RuntimeCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = runtime_case
    call_id = await add_call(case, revision=case.tool.approval_revision())
    entered = asyncio.Event()

    async def remote(*args: Any) -> dict[str, Any]:
        case.client.calls += 1
        entered.set()
        await asyncio.Event().wait()
        return {"content": []}

    monkeypatch.setattr(case.client, "call_tool", remote)
    task = asyncio.create_task(execute(case, call_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    rows = await audit_rows(case)
    assert [row.status for row in rows] == ["started", "cancelled"]
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert await case.supervisor._resume_tool_calls(db, session) == "processed"
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.status == "error"
        assert "not retried" in call.result["error"]
    assert case.client.calls == 1
