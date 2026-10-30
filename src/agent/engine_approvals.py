"""Request-local native decisions, shared by API processes and the fenced runner."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.engine_access import check_engine_access
from agent.events import add_event, publish
from agent.models import Approval, Session, ToolCall


def expired(moment: datetime | None) -> bool:
    return moment is None or moment.replace(tzinfo=UTC) <= datetime.now(UTC)


async def resolve_native(
    db: AsyncSession,
    settings: Settings,
    session: Session,
    approval_id: str,
    decision: str,
    comment: str | None,
    actor_user_id: str | None,
    generation: int | None,
    executor: str | None,
) -> str:
    # Acquire the same row lock as Stop and the runner before inspecting the record.
    await db.execute(
        update(Session).where(Session.id == session.id).values(owner_token=Session.owner_token)
    )
    await db.refresh(session)
    await check_engine_access(db, settings, session, actor_user_id)
    approval = await db.get(Approval, approval_id, populate_existing=True)
    if approval is None:
        raise LookupError("Approval not found")
    call = await db.get(ToolCall, approval.tool_call_id)
    if (
        decision not in {"approve", "reject"}
        or approval.status != "pending"
        or approval.executor != "engine"
        or executor != "engine"
        or generation != approval.generation
        or generation != session.run_generation
        or call is None
        or call.executor != "engine"
        or call.session_id != session.id
        or not call.native_request_id
        or call.status != "awaiting_confirmation"
        or session.stop_requested
        or session.status != "awaiting_confirmation"
        or not session.owner_token
        or expired(session.lease_until)
        or expired(approval.expires_at)
    ):
        raise ValueError("Native approval is expired, resolved or belongs to another run")
    claimed = cast(
        CursorResult[Any],
        await db.execute(
            update(Approval)
            .where(
                Approval.id == approval.id,
                Approval.status == "pending",
                Approval.delivered_at.is_(None),
            )
            .values(
                status="approved" if decision == "approve" else "rejected",
                comment=comment,
                resolved_at=datetime.now(UTC),
            )
            .execution_options(synchronize_session=False)
        ),
    )
    if claimed.rowcount != 1:
        raise ValueError("Native approval is already resolved")
    pending = [
        await add_event(
            db,
            "approval.resolved",
            {
                "approval_id": approval.id,
                "decision": decision,
                "generation": generation,
                "executor": "engine",
                "scope": approval.scope,
            },
            session.id,
        )
    ]
    await db.commit()
    await publish(pending)
    return session.id
