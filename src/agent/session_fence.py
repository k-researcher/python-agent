"""Generation and owner fencing for session execution.

A session run belongs to one executor at a time. The executor holds a fence: the run
generation, the owner epoch and a random owner token, plus a time-limited lease. Every
write of the executor checks the fence in the same transaction. If the fence does not
match, the write fails and the transaction rolls back, so a stale executor (after a
reclaim, a Stop or a new run) cannot change the session, its messages, events, tool
calls or audit rows.

Rules:
- A new run (start or new message) increments the generation and clears the owner.
- The first Stop increments the generation; a repeated Stop does not change it.
- An approval or the return of a child session continues the current generation.
- An expired lease permits neither writes nor renewal; a new owner gets a new epoch.

Limit: fencing protects database state only. An external effect of a tool (a file
write, a shell command, a network request) that already happened is not undone.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import ORMExecuteState, SessionTransaction
from sqlalchemy.orm import Session as SyncSession

from agent.models import Session as SessionRow
from agent.models import SessionRunOutcome

TERMINAL_STATUSES = ("completed", "stopped", "error", "interrupted")

# The full sessions table, so guarded updates can set any session column.
_sessions = cast(sa.Table, SessionRow.__table__)


class FenceLostError(RuntimeError):
    """The executor no longer owns the session run; stop all work of this run."""


@dataclass(frozen=True, slots=True)
class RunFence:
    session_id: str
    generation: int
    epoch: int
    token: str


def _now() -> datetime:
    return datetime.now(UTC)


def _unfinished() -> sa.ColumnElement[bool]:
    return sa.and_(
        _sessions.c.status.not_in(TERMINAL_STATUSES),
        ~sa.exists(
            sa.select(SessionRunOutcome.session_id).where(
                SessionRunOutcome.session_id == _sessions.c.id,
                SessionRunOutcome.generation == _sessions.c.run_generation,
            )
        ),
    )


def _owned(fence: RunFence, now: datetime, *, finishing: bool = False) -> sa.ColumnElement[bool]:
    """Return the condition that the fence still owns an unexpired lease."""
    return sa.and_(
        _sessions.c.id == fence.session_id,
        _sessions.c.run_generation == fence.generation,
        _sessions.c.owner_epoch == fence.epoch,
        _sessions.c.owner_token == fence.token,
        _sessions.c.lease_until >= now,
        sa.true() if finishing else _unfinished(),
    )


async def _rowcount(db: AsyncSession, statement: sa.Executable) -> int:
    result = cast(CursorResult[Any], await db.execute(statement))
    return result.rowcount


async def claim(
    db: AsyncSession,
    session_id: str,
    *,
    lease_seconds: float,
    now: datetime | None = None,
    expected_generation: int | None = None,
) -> RunFence | None:
    """Take the session run for this executor and commit; return None when it is not free.

    The run is free when it has no outcome, the session is not terminal, no Stop is pending,
    and no owner holds an unexpired lease. The claim increments the epoch, not the generation.
    With ``expected_generation``, the claim also requires that generation, in the same
    conditional update: a run scheduled for an older generation cannot take a newer one.
    """
    moment = now or _now()
    token = str(uuid.uuid4())
    conditions = [
        _sessions.c.id == session_id,
        _unfinished(),
        _sessions.c.stop_requested.is_(False),
        sa.or_(_sessions.c.owner_token.is_(None), _sessions.c.lease_until < moment),
    ]
    if expected_generation is not None:
        conditions.append(_sessions.c.run_generation == expected_generation)
    statement = (
        sa.update(_sessions)
        .where(*conditions)
        .values(
            owner_epoch=_sessions.c.owner_epoch + 1,
            owner_token=token,
            lease_until=moment + timedelta(seconds=lease_seconds),
        )
    )
    if await _rowcount(db, statement) != 1:
        await db.rollback()
        return None
    row = (
        await db.execute(
            sa.select(_sessions.c.run_generation, _sessions.c.owner_epoch).where(
                _sessions.c.id == session_id, _sessions.c.owner_token == token
            )
        )
    ).one()
    await db.commit()
    return RunFence(session_id, int(row.run_generation), int(row.owner_epoch), token)


async def renew(
    db: AsyncSession, fence: RunFence, *, lease_seconds: float, now: datetime | None = None
) -> bool:
    """Extend the lease and commit; return False when the fence is lost or the lease expired."""
    moment = now or _now()
    statement = (
        sa.update(_sessions)
        .where(_owned(fence, moment))
        .values(lease_until=moment + timedelta(seconds=lease_seconds))
    )
    renewed = await _rowcount(db, statement) == 1
    await (db.commit() if renewed else db.rollback())
    return renewed


async def release(db: AsyncSession, fence: RunFence) -> None:
    """Give the run back and commit; a lost fence releases nothing."""
    statement = (
        sa.update(_sessions)
        .where(
            _sessions.c.id == fence.session_id,
            _sessions.c.owner_epoch == fence.epoch,
            _sessions.c.owner_token == fence.token,
        )
        .values(owner_token=None, lease_until=None)
    )
    await _rowcount(db, statement)
    await db.commit()


async def fence_check(db: AsyncSession, fence: RunFence, *, now: datetime | None = None) -> None:
    """Verify the fence inside the current transaction; raise FenceLostError when it is lost.

    Call it in every executor transaction before the commit. The conditional update locks the
    session row until the commit, so a concurrent Stop or reclaim waits or wins cleanly. The
    caller must roll back on FenceLostError.
    """
    moment = now or _now()
    statement = (
        sa.update(_sessions)
        .where(
            _owned(fence, moment, finishing=db.sync_session.info.get("finishing_fence") == fence)
        )
        .values(owner_token=_sessions.c.owner_token)
    )
    if await _rowcount(db, statement) != 1:
        raise FenceLostError(f"Session {fence.session_id} run is no longer owned")


async def guarded_update(
    db: AsyncSession, fence: RunFence, values: dict[str, Any], *, now: datetime | None = None
) -> None:
    """Update session columns only while the fence owns the run; no commit."""
    moment = now or _now()
    statement = (
        sa.update(_sessions)
        .where(
            _owned(fence, moment, finishing=db.sync_session.info.get("finishing_fence") == fence)
        )
        .values(**values)
    )
    if await _rowcount(db, statement) != 1:
        raise FenceLostError(f"Session {fence.session_id} run is no longer owned")


async def new_generation(db: AsyncSession, session_id: str) -> None:
    """Start a new run generation and clear the owner; no commit."""
    await db.execute(
        sa.update(_sessions)
        .where(_sessions.c.id == session_id)
        .values(
            run_generation=_sessions.c.run_generation + 1,
            owner_token=None,
            lease_until=None,
            stop_requested=False,
        )
    )


async def invalidate_generation(
    db: AsyncSession, session_id: str, expected_generation: int | None = None
) -> tuple[int, str] | None:
    """Lock even a no-op until commit, then invalidate once and return the old run."""
    lock = (
        sa.update(_sessions)
        .where(_sessions.c.id == session_id)
        .values(owner_token=_sessions.c.owner_token)
    )
    if await _rowcount(db, lock) != 1:
        return None
    conditions = [_sessions.c.id == session_id, _sessions.c.stop_requested.is_(False)]
    if expected_generation is not None:
        conditions.extend(
            [
                _sessions.c.run_generation == expected_generation,
                _sessions.c.status.not_in(TERMINAL_STATUSES),
            ]
        )
    statement = (
        sa.update(_sessions)
        .where(*conditions)
        .values(
            run_generation=_sessions.c.run_generation + 1,
            owner_token=None,
            lease_until=None,
            stop_requested=True,
        )
        .returning(_sessions.c.run_generation, _sessions.c.status)
    )
    row = (await db.execute(statement)).one_or_none()
    return (int(row.run_generation) - 1, str(row.status)) if row is not None else None


async def stop_generation(db: AsyncSession, session_id: str) -> bool:
    """Invalidate the current run once. No commit."""
    return await invalidate_generation(db, session_id) is not None


async def stop_generation_scoped(
    db: AsyncSession, session_id: str, expected_generation: int
) -> bool:
    """Stop only the expected run generation; return False otherwise. No commit.

    The stop invalidates the run only when the session is still at the expected
    generation, no Stop is pending, and the session is not terminal. A late stop
    after a newer generation started does nothing. The check is only in the WHERE.
    """
    return await invalidate_generation(db, session_id, expected_generation) is not None


# The fence of the run that the current task executes. Tasks that the run starts inherit it.
CURRENT_FENCE: ContextVar[RunFence | None] = ContextVar("current_fence", default=None)
_guard_installed = False


def _check_fence(session: SyncSession, *, committing: bool = False) -> None:
    """Lock the owned run; at commit also require the exact finished status."""
    fence = CURRENT_FENCE.get()
    if fence is None or session.info.get("fence_exempt"):
        return
    finishing = session.info.get("finishing_fence") == fence
    conditions = [_owned(fence, _now(), finishing=finishing)]
    if committing and finishing:
        conditions.append(_sessions.c.status == session.info["finishing_run"][2])
    statement = sa.update(_sessions).where(*conditions).values(owner_token=_sessions.c.owner_token)
    result = session.connection().execute(statement)
    if result.rowcount != 1:
        raise FenceLostError(f"Session {fence.session_id} run is no longer owned")


def _check_before_commit(session: SyncSession) -> None:
    """Check database state after the commit-time flush, not the identity map."""
    if CURRENT_FENCE.get() is None or session.info.get("fence_exempt"):
        return
    session.flush()
    _check_fence(session, committing=True)


def _check_before_flush(session: SyncSession, flush_context: Any, instances: Any) -> None:
    fence = CURRENT_FENCE.get()
    if fence is None or session.info.get("fence_exempt"):
        return
    finishing_run = session.info.get("finishing_run")
    for row in session.dirty.union(session.new):
        if (
            isinstance(row, SessionRow)
            and sa.inspect(row).attrs.status.history.has_changes()
            and (
                row.status in TERMINAL_STATUSES
                or (finishing_run is not None and row.id == finishing_run[0])
            )
            and (
                session.info.get("finishing_fence") != fence
                or session.info.get("finishing_run") != (row.id, row.run_generation, row.status)
            )
        ):
            raise FenceLostError("Session status changes must match finish_run in this transaction")
    _check_fence(session)


def _check_before_dml(state: ORMExecuteState) -> None:
    if state.is_insert or state.is_update or state.is_delete:
        _check_fence(state.session)


def _clear_finish_permission(session: SyncSession) -> None:
    if session.in_nested_transaction():
        return
    session.info.pop("finishing_fence", None)
    session.info.pop("finishing_run", None)
    session.info.pop("finishing_savepoints", None)


def _remember_savepoint_finish(session: SyncSession, transaction: SessionTransaction) -> None:
    if transaction.nested:
        session.info.setdefault("finishing_savepoints", {})[transaction] = {
            key: session.info[key]
            for key in ("finishing_fence", "finishing_run")
            if key in session.info
        }


def _restore_savepoint_finish(session: SyncSession, transaction: SessionTransaction) -> None:
    if transaction.nested:
        previous = session.info.get("finishing_savepoints", {}).pop(transaction, None)
        if previous is not None:
            session.info.pop("finishing_fence", None)
            session.info.pop("finishing_run", None)
            session.info.update(previous)


def _clear_finished_transaction(session: SyncSession, transaction: SessionTransaction) -> None:
    if transaction.parent is None:
        _clear_finish_permission(session)


def install_commit_guard() -> None:
    """Register the commit guard once per process.

    While CURRENT_FENCE is set, flushes and explicit DML acquire the session fence
    before locking dependent rows. After flushing, every commit rechecks ownership
    and the finished database status. ORM status changes must match finish_run;
    savepoint rollback restores the enclosing transaction's finish permission.
    """
    global _guard_installed
    if not _guard_installed:
        event.listen(SyncSession, "before_flush", _check_before_flush)
        event.listen(SyncSession, "do_orm_execute", _check_before_dml)
        event.listen(SyncSession, "before_commit", _check_before_commit)
        event.listen(SyncSession, "after_transaction_create", _remember_savepoint_finish)
        event.listen(SyncSession, "after_commit", _clear_finish_permission)
        event.listen(SyncSession, "after_rollback", _clear_finish_permission)
        event.listen(SyncSession, "after_soft_rollback", _restore_savepoint_finish)
        event.listen(SyncSession, "after_transaction_end", _clear_finished_transaction)
        _guard_installed = True
