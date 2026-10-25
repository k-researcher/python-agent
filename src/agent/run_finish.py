"""Atomic terminal state, immutable outcome and durable events."""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.events import PendingEvent, add_event
from agent.models import Session, SessionRunOutcome
from agent.session_fence import (
    CURRENT_FENCE,
    TERMINAL_STATUSES,
    FenceLostError,
    invalidate_generation,
)

FinishSource = Literal["run", "dispatch", "stop", "scoped_stop", "recovery"]


async def finish_run(
    db: AsyncSession,
    session_id: str,
    generation: int | None,
    status: str,
    *,
    source: FinishSource = "run",
    reason: str | None = None,
    error: str | None = None,
) -> list[PendingEvent]:
    """Record the first outcome and its events; the caller commits and publishes."""
    if status not in TERMINAL_STATUSES:
        raise ValueError("A run outcome must be terminal")
    if generation is None and source != "stop":
        raise ValueError("A run generation is required")
    fence = CURRENT_FENCE.get()
    if (
        source == "run"
        and fence is not None
        and (fence.session_id != session_id or fence.generation != generation)
    ):
        raise FenceLostError("The fence does not belong to this run")
    if source in {"stop", "scoped_stop"}:
        if status != "stopped":
            raise ValueError("Stop must record a stopped status")
        stopped = await invalidate_generation(
            db, session_id, generation if source == "scoped_stop" else None
        )
        if stopped is None:
            return []
        generation, previous_status = stopped
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=status)
            .execution_options(synchronize_session=False)
        )
        if previous_status in TERMINAL_STATUSES:
            if previous_status == status:
                return []
            return [
                await add_event(
                    db, "session.status", {"status": status, "generation": generation}, session_id
                )
            ]
        existing = await db.get(SessionRunOutcome, (session_id, generation))
        if existing is not None:
            return []
    else:
        conditions = [
            Session.id == session_id,
            Session.run_generation == generation,
            Session.status.not_in(TERMINAL_STATUSES),
            Session.stop_requested.is_(False),
            ~exists(
                select(SessionRunOutcome.session_id).where(
                    SessionRunOutcome.session_id == Session.id,
                    SessionRunOutcome.generation == generation,
                )
            ),
        ]
        values: dict[str, Any] = {"status": status, "error": error}
        if source == "dispatch":
            conditions.extend([Session.status == "pending", Session.owner_token.is_(None)])
        if source == "recovery":
            if status != "interrupted":
                raise ValueError("Recovery must record an interrupted status")
            conditions.append(Session.status == "running")
            values.update(
                run_generation=Session.run_generation + 1, owner_token=None, lease_until=None
            )
        result = await db.execute(
            update(Session)
            .where(*conditions)
            .values(**values)
            .returning(Session.id)
            .execution_options(synchronize_session=False)
        )
        if result.one_or_none() is None:
            return []
    assert generation is not None
    db.add(
        SessionRunOutcome(
            session_id=session_id,
            generation=generation,
            status=status,
            reason=reason,
            error=error,
        )
    )
    if (
        source == "run"
        and fence is not None
        and fence.session_id == session_id
        and fence.generation == generation
    ):
        db.sync_session.info["finishing_fence"] = fence
        db.sync_session.info["finishing_run"] = (session_id, generation, status)
    payload: dict[str, Any] = {"status": status, "generation": generation}
    if reason is not None:
        payload["reason"] = reason
    if error is not None:
        payload["error"] = error
    event_type = "session.error" if status == "error" else "session.status"
    return [await add_event(db, event_type, payload, session_id)]
