from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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


async def test_thirty_concurrent_terminals_keep_one_calibration_stable(tmp_path: Path) -> None:
    session_id, key = await setup_audit(tmp_path)
    started = [await start(session_id, key) for _ in range(30)]
    terminal = [
        (audit_id, replace(record, status="completed", prompt_tokens=200, completion_tokens=3))
        for audit_id, record in started
    ]
    await asyncio.gather(*(finish_attempt(audit_id, record, key) for audit_id, record in terminal))
    async with session_factory() as db:
        rows = list(
            await db.scalars(select(OutboundAudit).where(OutboundAudit.session_id == session_id))
        )
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
    assert len(rows) == 30
    assert all(row.status == "completed" and row.finished_at is not None for row in rows)
    assert all((row.payload_bytes, row.payload_sha256) == (321, "a" * 64) for row in rows)
    assert all(row.prompt_tokens == 200 and row.completion_tokens == 3 for row in rows)
    assert calibration is not None
    assert calibration.samples == 30
    await asyncio.gather(*(finish_attempt(audit_id, record, key) for audit_id, record in terminal))
    await asyncio.gather(*(cancel_attempt(audit_id) for audit_id, _ in started))
    async with session_factory() as db:
        rows = list(
            await db.scalars(select(OutboundAudit).where(OutboundAudit.session_id == session_id))
        )
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
    assert len(rows) == 30
    assert all(row.status == "completed" and row.finished_at is not None for row in rows)
    assert calibration is not None
    assert calibration.samples == 30


async def test_calibration_conflict_rolls_back_both_and_can_retry_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, key = await setup_audit(tmp_path)
    audit_id, initial = await start(session_id, key)
    async with session_factory() as db:
        db.add(
            TokenCalibration(
                provider=key.provider,
                model=key.model,
                wire_version=key.wire_version,
                factor=1.0,
                samples=1,
            )
        )
        await db.commit()

    real_factory = session_factory
    updates = 0
    locked_queries = []

    def patched_factory():
        real = real_factory()
        original_execute = real.execute
        original_scalar = real.scalar

        async def patched_execute(statement, *args, **kwargs):
            nonlocal updates
            table = getattr(statement, "table", None)
            if getattr(statement, "is_update", False) and getattr(table, "name", None) == (
                TokenCalibration.__table__.name
            ):
                updates += 1
                return SimpleNamespace(rowcount=0)
            return await original_execute(statement, *args, **kwargs)

        async def patched_scalar(statement, *args, **kwargs):
            from sqlalchemy.dialects import postgresql

            if getattr(statement, "is_select", False):
                locked_queries.append(str(statement.compile(dialect=postgresql.dialect())))
            return await original_scalar(statement, *args, **kwargs)

        monkeypatch.setattr(real, "execute", patched_execute)
        monkeypatch.setattr(real, "scalar", patched_scalar)
        return real

    terminal = replace(initial, status="completed", prompt_tokens=200, completion_tokens=3)
    with monkeypatch.context() as scoped:
        scoped.setattr("agent.attempt_audit.session_factory", patched_factory)
        with pytest.raises(RuntimeError, match="Token calibration update conflict"):
            await finish_attempt(audit_id, terminal, key)

    assert updates == 5
    assert locked_queries and all("FOR UPDATE" in query for query in locked_queries)
    async with session_factory() as db:
        audit = await db.get(OutboundAudit, audit_id)
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
        assert audit is not None and audit.status == "started" and audit.finished_at is None
        assert calibration is not None
        assert (calibration.samples, calibration.factor) == (1, 1.0)

    await finish_attempt(audit_id, terminal, key)
    await finish_attempt(audit_id, terminal, key)
    async with session_factory() as db:
        audit = await db.get(OutboundAudit, audit_id)
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
        assert audit is not None and audit.status == "completed" and audit.finished_at is not None
        assert audit.prompt_tokens == 200
        assert calibration is not None
        assert calibration.samples == 2
        assert calibration.factor == pytest.approx(1.2)
