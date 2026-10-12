from __future__ import annotations

from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError

from agent.database import session_factory
from agent.llm import AttemptRecord
from agent.models import OutboundAudit, TokenCalibration, utcnow
from agent.token_calibration import ALPHA, MAX_FACTOR, MIN_FACTOR, CalibrationKey


async def start_attempt(session_id: str, fields: dict[str, Any], record: AttemptRecord) -> int:
    async with session_factory() as db:
        audit = OutboundAudit(
            session_id=session_id,
            category="llm",
            status="started",
            payload_bytes=record.payload_bytes,
            payload_sha256=record.payload_sha256,
            **fields,
        )
        db.add(audit)
        await db.commit()
        return audit.id


async def finish_attempt(audit_id: int, record: AttemptRecord, key: CalibrationKey) -> None:
    async with session_factory() as db:
        status = "cancelled" if record.error_kind == "cancelled" else record.status
        transitioned = cast(
            CursorResult[Any],
            await db.execute(
                update(OutboundAudit)
                .where(
                    OutboundAudit.id == audit_id,
                    OutboundAudit.status == "started",
                )
                .values(
                    status=status,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                    error_kind=record.error_kind,
                    finished_at=utcnow(),
                )
            ),
        )
        if not transitioned.rowcount:
            return
        audit = await db.get(OutboundAudit, audit_id)
        assert audit is not None
        raw = audit.raw_estimate or 0
        actual = record.prompt_tokens
        if record.status == "completed" and raw > 0 and actual is not None and actual > 0:
            for _attempt in range(5):
                row = await db.scalar(
                    select(TokenCalibration)
                    .where(
                        TokenCalibration.provider == key.provider,
                        TokenCalibration.model == key.model,
                        TokenCalibration.wire_version == key.wire_version,
                    )
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                old_factor = row.factor if row is not None else 1.0
                samples = row.samples if row is not None else 0
                factor = min(
                    MAX_FACTOR,
                    max(
                        MIN_FACTOR,
                        (1 - ALPHA) * old_factor + ALPHA * actual / raw,
                    ),
                )
                if row is None:
                    try:
                        async with db.begin_nested():
                            db.add(
                                TokenCalibration(
                                    provider=key.provider,
                                    model=key.model,
                                    wire_version=key.wire_version,
                                    factor=factor,
                                    samples=1,
                                )
                            )
                            await db.flush()
                        break
                    except IntegrityError:
                        continue
                result = cast(
                    CursorResult[Any],
                    await db.execute(
                        update(TokenCalibration)
                        .where(
                            TokenCalibration.provider == key.provider,
                            TokenCalibration.model == key.model,
                            TokenCalibration.wire_version == key.wire_version,
                            TokenCalibration.samples == samples,
                        )
                        .values(factor=factor, samples=samples + 1, updated_at=utcnow())
                    ),
                )
                if result.rowcount:
                    break
            else:
                raise RuntimeError("Token calibration update conflict")
        await db.commit()


async def cancel_attempt(audit_id: int) -> None:
    async with session_factory() as db:
        await db.execute(
            update(OutboundAudit)
            .where(
                OutboundAudit.id == audit_id,
                OutboundAudit.status.in_({"started", "error"}),
            )
            .values(status="cancelled", error_kind="cancelled", finished_at=utcnow())
        )
        await db.commit()
