from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, cast

from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.checkpoints import load_context_rows, save_checkpoint, source_item
from agent.context_blocks import split_blocks
from agent.context_projection import HistorySnapshot, SourceMessage
from agent.context_summary import SummaryCheckpoint, source_hash
from agent.database import session_factory
from agent.models import Message, Session, SessionStatus


class HistoryChanged(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedHistory:
    snapshot: HistorySnapshot
    cursor: int
    fingerprint: str
    covered_messages: int


async def load_history(db: AsyncSession, session_id: str) -> LoadedHistory:
    rows = await load_context_rows(db, session_id)
    summary = next((row for row in rows if row.kind == "summary"), None)
    sources = [row for row in rows if row.kind != "summary"]
    checkpoint = None
    covered_messages = 0
    if summary is not None:
        checkpoint = SummaryCheckpoint(
            summary.content or "",
            summary.covers_until or 0,
            summary.source_hash or "",
            summary.summary_version or 0,
        )
        covered_messages = int(
            await db.scalar(
                select(func.count(Message.id)).where(
                    Message.session_id == session_id,
                    Message.kind == "normal",
                    Message.skipped.is_(False),
                    Message.id <= checkpoint.covers_until,
                )
            )
            or 0
        )
        covered_messages -= sum(
            row.kind == "normal" and not row.skipped and row.id <= checkpoint.covers_until
            for row in sources
        )
    cursor = int(
        await db.scalar(
            select(func.max(Message.id)).where(
                Message.session_id == session_id,
                Message.kind != "summary",
            )
        )
        or 0
    )
    snapshot = HistorySnapshot(
        tuple(
            SourceMessage.from_row(row)
            for row in sources
            if row.kind == "normal" and not row.skipped
        ),
        checkpoint,
        summary.id if summary is not None else None,
    )
    return LoadedHistory(
        snapshot,
        cursor,
        source_hash([source_item(row) for row in sources]),
        max(0, covered_messages),
    )


async def ensure_active(session_id: str) -> None:
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        if session is None or session.stop_requested or session.status != "running":
            raise asyncio.CancelledError


async def persist_checkpoint(
    session_id: str,
    history: LoadedHistory,
    checkpoint: SummaryCheckpoint,
    model_id: str | None,
) -> int:
    async with session_factory() as db:
        claimed = cast(
            CursorResult[Any],
            await db.execute(
                update(Session)
                .where(
                    Session.id == session_id,
                    Session.status == SessionStatus.running.value,
                    Session.stop_requested.is_(False),
                )
                .values(status=SessionStatus.running.value)
            ),
        )
        if not claimed.rowcount:
            raise asyncio.CancelledError
        current = await load_history(db, session_id)
        if (
            current.cursor != history.cursor
            or current.fingerprint != history.fingerprint
            or current.snapshot.checkpoint_message_id != history.snapshot.checkpoint_message_id
            or current.snapshot.checkpoint != history.snapshot.checkpoint
        ):
            raise HistoryChanged("Context source changed during preparation")
        prefix: list[dict[str, Any]] = []
        cursor = 0
        while True:
            page = list(
                await db.scalars(
                    select(Message)
                    .where(
                        Message.session_id == session_id,
                        Message.kind != "summary",
                        Message.id > cursor,
                        Message.id <= checkpoint.covers_until,
                    )
                    .order_by(Message.id)
                    .limit(200)
                )
            )
            if not page:
                break
            prefix.extend(source_item(row) for row in page)
            cursor = page[-1].id
        if cursor != checkpoint.covers_until:
            raise HistoryChanged("Checkpoint boundary is not a source message")
        blocks = split_blocks(
            [item for item in prefix if item["kind"] == "normal" and not item["skipped"]]
        )
        if not blocks or not blocks[-1].complete or blocks[-1].kind in {"system", "user"}:
            raise HistoryChanged("Checkpoint boundary is not a complete exchange")
        canonical = replace(checkpoint, source_hash=source_hash(prefix))
        return await save_checkpoint(db, session_id, canonical, model_id)
