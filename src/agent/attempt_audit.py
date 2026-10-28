from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError

from agent.database import session_factory
from agent.events import add_event, publish
from agent.inference_guard import CURRENT_GUARD, terminal_reason
from agent.llm import AttemptRecord
from agent.model_config import PricingSpec
from agent.models import OutboundAudit, TokenCalibration, utcnow
from agent.token_calibration import ALPHA, MAX_FACTOR, MIN_FACTOR, CalibrationKey
from agent.usage import UsageRecord, exact_cost, valid_token_count


async def start_attempt(
    session_id: str,
    fields: dict[str, Any],
    record: AttemptRecord,
    *,
    pricing: PricingSpec | None = None,
) -> int:
    async with session_factory() as db:
        audit = OutboundAudit(
            session_id=session_id,
            category="llm",
            status="started",
            payload_bytes=record.payload_bytes,
            payload_sha256=record.payload_sha256,
            pricing_snapshot=pricing.model_dump() if pricing is not None else None,
            **fields,
        )
        db.add(audit)
        await db.flush()
        pending = await add_event(
            db, "outbound.audit", {"audit_id": audit.id, "status": "started"}, session_id
        )
        await db.commit()
        await publish([pending])
        return audit.id


async def finish_attempt(audit_id: int, record: AttemptRecord, key: CalibrationKey) -> None:
    async with session_factory() as db:
        audit = await db.get(OutboundAudit, audit_id)
        record = _terminal_record(record)
        status = "cancelled" if record.error_kind == "cancelled" else record.status
        if audit is None or audit.status != "started":
            return
        prompt_tokens = valid_token_count(record.prompt_tokens)
        completion_tokens = valid_token_count(record.completion_tokens)
        try:
            pricing = (
                PricingSpec.model_validate(audit.pricing_snapshot)
                if audit.pricing_snapshot is not None
                else None
            )
        except ValidationError:
            pricing = None
        amount = exact_cost(UsageRecord(prompt_tokens, completion_tokens), pricing)
        cost_amount = str(amount) if amount is not None else None
        if cost_amount is not None and len(cost_amount) > 100:
            cost_amount = None
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
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_amount=cost_amount,
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
        actual = prompt_tokens
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
        final_record = _terminal_record(record)
        if final_record != record:
            await db.rollback()
            await finish_attempt(audit_id, final_record, key)
            return
        pending = await add_event(
            db, "outbound.audit", {"audit_id": audit_id, "status": status}, audit.session_id
        )
        await db.commit()
        await publish([pending])


def _terminal_record(record: AttemptRecord) -> AttemptRecord:
    guard = CURRENT_GUARD.get()
    if guard is None or not guard.closed:
        return record
    reason = guard.reason
    kind = terminal_reason(reason)
    return replace(record, status="error", error_kind=kind)


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
