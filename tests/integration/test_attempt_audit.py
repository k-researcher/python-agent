from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select

from agent.attempt_audit import cancel_attempt, finish_attempt, start_attempt
from agent.database import init_database, session_factory
from agent.llm import AttemptRecord
from agent.models import OutboundAudit, Project, Session, TokenCalibration
from agent.token_calibration import CalibrationKey


async def setup_audit(root: Path) -> tuple[str, CalibrationKey]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name=root.name, root_path=str(root))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="audit", status="running")
        db.add(session)
        await db.commit()
        return session.id, CalibrationKey("test", session.id, "wire-test")


async def start(session_id: str, key: CalibrationKey) -> tuple[int, AttemptRecord]:
    record = AttemptRecord(0, "started", 321, "a" * 64)
    audit_id = await start_attempt(
        session_id,
        {
            "destination": "https://llm.example/v1/chat/completions",
            "operation": "chat:test",
            "inference_id": session_id,
            "purpose": "inference",
            "provider": key.provider,
            "model_id": "test",
            "model": key.model,
            "wire_version": key.wire_version,
            "raw_estimate": 100,
            "factor": 1.0,
        },
        record,
    )
    return audit_id, record


async def test_terminal_audit_and_calibration_are_idempotent(tmp_path: Path) -> None:
    session_id, key = await setup_audit(tmp_path)
    audit_id, initial = await start(session_id, key)
    terminal = replace(initial, status="completed", prompt_tokens=200, completion_tokens=3)
    await asyncio.gather(*(finish_attempt(audit_id, terminal, key) for _ in range(8)))
    async with session_factory() as db:
        row = await db.get(OutboundAudit, audit_id)
        assert row is not None
        assert (
            row.status,
            row.payload_bytes,
            row.payload_sha256,
            row.prompt_tokens,
            row.completion_tokens,
        ) == ("completed", 321, "a" * 64, 200, 3)
        assert row.finished_at is not None
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
        assert calibration is not None
        assert calibration.samples == 1
        assert calibration.factor == pytest.approx(1.2)


async def test_concurrent_audits_do_not_lose_calibration_samples(tmp_path: Path) -> None:
    session_id, key = await setup_audit(tmp_path)
    started = [await start(session_id, key) for _ in range(10)]
    await asyncio.gather(
        *(
            finish_attempt(audit_id, replace(record, status="completed", prompt_tokens=100), key)
            for audit_id, record in started
        )
    )
    async with session_factory() as db:
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
        assert calibration is not None
        assert calibration.samples == 10
        assert calibration.factor == 1.0
        rows = list(
            await db.scalars(select(OutboundAudit).where(OutboundAudit.session_id == session_id))
        )
        assert len(rows) == 10
        assert all(row.status == "completed" and row.finished_at is not None for row in rows)


@pytest.mark.parametrize(
    "status,error_kind", [("completed", None), ("error", "network"), ("error", "cancelled")]
)
async def test_unknown_usage_never_becomes_zero_or_calibration(
    tmp_path: Path, status: str, error_kind: str | None
) -> None:
    session_id, key = await setup_audit(tmp_path)
    audit_id, initial = await start(session_id, key)
    await finish_attempt(audit_id, replace(initial, status=status, error_kind=error_kind), key)
    async with session_factory() as db:
        row = await db.get(OutboundAudit, audit_id)
        assert row is not None
        assert row.prompt_tokens is None and row.completion_tokens is None
        assert row.status == ("cancelled" if error_kind == "cancelled" else status)
        assert await db.get(TokenCalibration, (key.provider, key.model, key.wire_version)) is None


async def test_cancelled_audit_cannot_later_be_completed(tmp_path: Path) -> None:
    session_id, key = await setup_audit(tmp_path)
    audit_id, initial = await start(session_id, key)
    await cancel_attempt(audit_id)
    await finish_attempt(audit_id, replace(initial, status="completed", prompt_tokens=200), key)
    async with session_factory() as db:
        row = await db.get(OutboundAudit, audit_id)
        assert row is not None
        assert row.status == "cancelled" and row.error_kind == "cancelled"
        assert await db.get(TokenCalibration, (key.provider, key.model, key.wire_version)) is None
