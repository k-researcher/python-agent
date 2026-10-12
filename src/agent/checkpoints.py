"""Load session context rows and persist summary checkpoints."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.context_summary import SUMMARY_VERSION, SummaryCheckpoint
from agent.models import Message

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be saved."""


async def load_context_rows(db: AsyncSession, session_id: str) -> list[Message]:
    """Return session rows for the context window.

    With a usable checkpoint, anchor around it: leading system rows, the first
    normal user message, the checkpoint itself and the covered tail. Otherwise
    return all session rows ordered by id.
    """
    checkpoint = await _best_checkpoint(db, session_id)
    if checkpoint is None:
        all_rows = await db.scalars(
            select(Message).where(Message.session_id == session_id).order_by(Message.id)
        )
        return list(all_rows)

    covered = checkpoint.covers_until
    assert covered is not None
    first_user = await db.scalar(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.role == "user",
            Message.kind == "normal",
        )
        .order_by(Message.id)
        .limit(1)
    )
    system_rows = await _leading_system_rows(db, session_id)
    tail = await _tail_rows(db, session_id, covered)

    rows = [*system_rows]
    if first_user is not None:
        rows.append(first_user)
    rows.append(checkpoint)
    rows.extend(tail)
    # Preserve the projection order (anchors → checkpoint → tail): the checkpoint
    # is inserted after the messages it covers, so a global id sort would move it
    # behind newer rows. Drop duplicate ids keeping the first occurrence.
    return list({row.id: row for row in rows}.values())


async def save_checkpoint(
    db: AsyncSession,
    session_id: str,
    checkpoint: SummaryCheckpoint,
    model_id: str | None,
) -> int:
    """Validate the checkpoint, insert it and return its message id.

    Reuse the existing row when the unique checkpoint key already exists.
    """
    await _validate_checkpoint(db, session_id, checkpoint)
    message = Message(
        session_id=session_id,
        role="user",
        kind="summary",
        content=checkpoint.text,
        covers_until=checkpoint.covers_until,
        source_hash=checkpoint.source_hash,
        summary_version=checkpoint.version,
        summary_model=model_id,
    )
    db.add(message)
    try:
        await db.flush()
        checkpoint_id = message.id
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing_id = await db.scalar(
            select(Message.id).where(
                Message.session_id == session_id,
                Message.covers_until == checkpoint.covers_until,
                Message.source_hash == checkpoint.source_hash,
                Message.summary_version == checkpoint.version,
            )
        )
        if existing_id is None:
            raise
        return existing_id
    return checkpoint_id


async def _best_checkpoint(db: AsyncSession, session_id: str) -> Message | None:
    """Return the best usable checkpoint row, or None when none is usable.

    Candidates are scanned by decreasing coverage (most recent on top); an
    unusable row with a larger ``covers_until`` is skipped instead of winning
    the selection, because a stored row can carry malformed metadata even when
    the table constraints accepted it.
    """
    stmt = (
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.kind == "summary",
            Message.summary_version == SUMMARY_VERSION,
        )
        .order_by(Message.covers_until.desc(), Message.id.desc())
    )
    candidates = (await db.scalars(stmt)).all()
    for candidate in candidates:
        if await _is_usable_checkpoint(db, session_id, candidate):
            return candidate
    return None


async def _is_usable_checkpoint(db: AsyncSession, session_id: str, message: Message) -> bool:
    """Return True when a stored checkpoint row carries well-formed metadata.

    Rejects whitespace-only text, non-hex or wrong-length source hashes and
    coverage that does not point to a real source (non-summary) message of the
    same session. The covered message must be older than the checkpoint row: a
    checkpoint is always saved after the messages that it covers.
    """
    if not message.content or not message.content.strip():
        return False
    if not message.source_hash or not _HEX64.fullmatch(message.source_hash):
        return False
    if message.covers_until is None:
        return False
    covered = await db.scalar(
        select(Message.id).where(
            Message.id == message.covers_until,
            Message.session_id == session_id,
            Message.kind != "summary",
        )
    )
    return covered is not None and covered < message.id


async def _leading_system_rows(db: AsyncSession, session_id: str) -> list[Message]:
    """Return the initial contiguous system rows that stay in the context window.

    The prefix ends at the first message whose role is not ``system``, so a
    system row that follows an assistant or tool message is never treated as
    part of the leading anchor.
    """
    prefix_end = await db.scalar(
        select(Message.id)
        .where(
            Message.session_id == session_id,
            Message.role != "system",
        )
        .order_by(Message.id)
        .limit(1)
    )
    if prefix_end is None:
        return []
    rows = await db.scalars(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.role == "system",
            Message.kind == "normal",
            Message.id < prefix_end,
        )
        .order_by(Message.id)
    )
    return list(rows)


async def _tail_rows(db: AsyncSession, session_id: str, covered: int) -> list[Message]:
    """Return non-summary source rows that follow the checkpoint coverage."""
    rows = await db.scalars(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.id > covered,
            Message.kind != "summary",
        )
        .order_by(Message.id)
    )
    return list(rows)


async def _validate_checkpoint(
    db: AsyncSession,
    session_id: str,
    checkpoint: SummaryCheckpoint,
) -> None:
    """Check that the checkpoint metadata is eligible for storage."""
    if not checkpoint.text or not checkpoint.text.strip():
        raise CheckpointError("checkpoint text must not be empty")
    if not _HEX64.fullmatch(checkpoint.source_hash):
        raise CheckpointError("source_hash must be 64 hex characters")
    covered = await db.scalar(
        select(Message.id).where(
            Message.id == checkpoint.covers_until,
            Message.session_id == session_id,
            Message.kind != "summary",
        )
    )
    if covered is None:
        raise CheckpointError("covers_until must reference a message in this session")
