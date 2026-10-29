"""Durable external submissions, terminal bookkeeping and restart reconciliation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import JSON, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.events import PendingEvent, add_event
from agent.message_tree import append_message
from agent.models import Approval, EngineTurn, Message, Session, ToolCall

UNKNOWN_RESULT = "результат неизвестен, проверьте файлы"


async def finish_engine_turns(
    db: AsyncSession, session_id: str, generation: int, status: str, reason: str | None
) -> list[PendingEvent]:
    """Called only by the winning finish_run, in its transaction."""
    turns = list(
        await db.scalars(
            select(EngineTurn).where(
                EngineTurn.session_id == session_id,
                EngineTurn.generation == generation,
                EngineTurn.state != "done",
            )
        )
    )
    if not turns:
        return []
    pending = []
    approvals = list(
        await db.scalars(
            select(Approval)
            .join(ToolCall)
            .where(
                ToolCall.session_id == session_id,
                Approval.executor == "engine",
                Approval.generation == generation,
                Approval.delivered_at.is_(None),
                Approval.status.in_(["pending", "approved", "rejected"]),
            )
        )
    )
    for approval in approvals:
        approval.status, approval.resolved_at = "cancelled", datetime.now(UTC)
        pending.append(
            await add_event(
                db,
                "approval.cancelled",
                {
                    "approval_id": approval.id,
                    "generation": generation,
                    "executor": "engine",
                },
                session_id,
            )
        )
    for turn in turns:
        stream = (turn.last_item_ids or {}).get("_stream", {})
        if stream and not stream.get("closed"):
            pending.append(
                await add_event(
                    db,
                    "message.created",
                    {
                        "role": "assistant",
                        **{
                            key: stream[key]
                            for key in ("message_id", "generation", "stream_no", "stream_id")
                        },
                    },
                    session_id,
                )
            )
            turn.last_item_ids = {**turn.last_item_ids, "_stream": {**stream, "closed": True}}
        turn.state = "done"
        turn.outcome = {"status": status, "reason": reason}
    if status != "completed":
        await db.execute(
            update(Message)
            .where(
                Message.session_id == session_id,
                Message.run_generation == generation,
                Message.role == "assistant",
                or_(Message.tool_calls.is_(None), Message.tool_calls == JSON.NULL),
            )
            .values(kind="interrupted")
        )
    for call in await db.scalars(
        select(ToolCall).where(
            ToolCall.session_id == session_id,
            ToolCall.executor == "engine",
            ToolCall.status.in_(["pending", "running", "awaiting_confirmation"]),
        )
    ):
        call.status = "cancelled"
        call.result = {"success": False, "error": reason or status}
        message = await append_message(
            db,
            session_id,
            role="tool",
            tool_call_id=call.id,
            content=json.dumps(call.result, ensure_ascii=False),
        )
        pending.append(
            await add_event(
                db, "message.created", {"role": "tool", "message_id": message.id}, session_id
            )
        )
        pending.append(
            await add_event(
                db, "tool.completed", {"tool_call_id": call.id, "status": call.status}, session_id
            )
        )
    return pending


async def recover_engines(db: AsyncSession) -> list[PendingEvent]:
    """Never steal a live lease or dispatch an uncertain submission."""
    from agent.run_finish import finish_run

    now = datetime.now(UTC)
    free = or_(Session.owner_token.is_(None), Session.lease_until < now)
    sessions = list(
        await db.scalars(
            select(Session).where(
                Session.engine_profile.is_not(None),
                Session.status.in_(
                    ["pending", "running", "awaiting_confirmation", "awaiting_child"]
                ),
                Session.stop_requested.is_(False),
                free,
            )
        )
    )
    pending = []
    for session in sessions:
        turns = list(
            await db.scalars(
                select(EngineTurn).where(
                    EngineTurn.session_id == session.id,
                    EngineTurn.generation == session.run_generation,
                )
            )
        )
        if any(turn.state != "prepared" for turn in turns):
            pending.extend(
                await finish_run(
                    db,
                    session.id,
                    session.run_generation,
                    "interrupted",
                    source="recovery",
                    reason=UNKNOWN_RESULT,
                )
            )
        else:
            await db.execute(
                update(Session)
                .where(
                    Session.id == session.id,
                    Session.run_generation == session.run_generation,
                    Session.status.in_(
                        ["pending", "running", "awaiting_confirmation", "awaiting_child"]
                    ),
                    Session.stop_requested.is_(False),
                    free,
                )
                .values(status="pending", owner_token=None, lease_until=None)
                .execution_options(synchronize_session=False)
            )
    return pending
