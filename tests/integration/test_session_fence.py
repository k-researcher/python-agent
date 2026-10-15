"""Integration tests for session run fencing (src/agent/session_fence.py).

The migrated schema (embedded mode runs migrations to ``0009p``) carries the four
fencing columns: ``run_generation``, ``owner_epoch``, ``owner_token`` and
``lease_until``.  Each test verifies the exact state of the ``sessions`` row after
every operation.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from agent.database import init_database, session_factory
from agent.models import Message, Project, Session, SessionStatus
from agent.session_fence import (
    FenceLostError,
    RunFence,
    claim,
    fence_check,
    guarded_update,
    new_generation,
    release,
    renew,
    stop_generation,
    stop_generation_scoped,
)


async def new_session(root: Path, **values: Any) -> str:
    """Create a fresh project and session, returning the session id."""
    await init_database(recover_running=False)
    tag = uuid.uuid4().hex[:8]
    async with session_factory() as db:
        project = Project(name=f"fence-{tag}", root_path=str(root / tag))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="fence", configuration={}, **values)
        db.add(session)
        await db.commit()
        return session.id


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalise a SQLite-loaded datetime to a timezone-aware UTC value."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


async def fence_state(session_id: str) -> dict[str, Any]:
    """Read the full fencing-relevant state of one session row."""
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        return {
            "title": row.title,
            "status": row.status,
            "stop_requested": row.stop_requested,
            "run_generation": row.run_generation,
            "owner_epoch": row.owner_epoch,
            "owner_token": row.owner_token,
            "lease_until": _as_utc(row.lease_until),
        }


async def count_messages(session_id: str) -> int:
    async with session_factory() as db:
        total = await db.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
    return int(total or 0)


async def test_claim_free_session_sets_fence_and_second_claim_fails(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.pending.value)
    now = datetime.now(UTC)
    fence = None
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None
    assert fence.generation == 0
    assert fence.epoch == 1

    async with session_factory() as db:
        again = await claim(db, session_id, lease_seconds=60, now=now)
    assert again is None

    state = await fence_state(session_id)
    assert state["status"] == SessionStatus.pending.value
    assert state["stop_requested"] is False
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 1
    assert state["owner_token"] == fence.token
    assert state["lease_until"] is not None
    assert state["lease_until"] > now


async def test_expired_lease_new_owner_rejects_old_fence(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    t0 = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=t0)
    assert fence is not None
    assert fence.generation == 0 and fence.epoch == 1

    expired = t0 + timedelta(seconds=61)

    async with session_factory() as db:
        assert await renew(db, fence, lease_seconds=60, now=expired) is False
    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=expired)
    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await guarded_update(db, fence, {"status": "error"}, now=expired)

    async with session_factory() as db:
        new_fence = await claim(db, session_id, lease_seconds=60, now=expired)
    assert new_fence is not None
    assert new_fence.generation == 0
    assert new_fence.epoch == 2

    state = await fence_state(session_id)
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 2
    assert state["owner_token"] == new_fence.token
    assert state["lease_until"] is not None
    assert state["lease_until"] > expired


async def test_expired_lease_without_new_owner_blocks_old_fence(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    t0 = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=t0)
    assert fence is not None
    expired = t0 + timedelta(seconds=61)

    async with session_factory() as db:
        assert await renew(db, fence, lease_seconds=60, now=expired) is False
    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=expired)

    state = await fence_state(session_id)
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 1
    assert state["owner_token"] == fence.token
    assert state["lease_until"] is not None
    assert state["lease_until"] < expired


async def test_guarded_update_can_set_any_session_column(tmp_path: Path) -> None:
    """A guarded update sets any session column while the fence owns the run."""
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None

    async with session_factory() as db:
        await guarded_update(db, fence, {"title": "changed", "error": "boom"}, now=now)
        await db.commit()
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None and row.title == "changed" and row.error == "boom"


@pytest.mark.parametrize("status", ["completed", "stopped", "error"])
async def test_claim_denied_for_terminal_status(tmp_path: Path, status: str) -> None:
    session_id = await new_session(tmp_path, status=status)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is None
    state = await fence_state(session_id)
    assert state["status"] == status
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 0
    assert state["owner_token"] is None
    assert state["lease_until"] is None


async def test_claim_denied_when_stop_requested(tmp_path: Path) -> None:
    session_id = await new_session(
        tmp_path, status=SessionStatus.running.value, stop_requested=True
    )
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is None
    state = await fence_state(session_id)
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 0
    assert state["owner_token"] is None
    assert state["lease_until"] is None


async def test_stop_generation_invalidates_run_once(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None

    async with session_factory() as db:
        first = await stop_generation(db, session_id)
        await db.commit()
    assert first is True
    state1 = await fence_state(session_id)
    assert state1["run_generation"] == 1
    assert state1["stop_requested"] is True
    assert state1["owner_token"] is None
    assert state1["lease_until"] is None
    assert state1["owner_epoch"] == 1

    async with session_factory() as db:
        second = await stop_generation(db, session_id)
        await db.commit()
    assert second is False
    state2 = await fence_state(session_id)
    assert state2["run_generation"] == 1
    assert state2["stop_requested"] is True

    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=now)


async def test_new_generation_clears_owner_and_old_fence_cannot_write(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None

    async with session_factory() as db:
        await new_generation(db, session_id)
        await db.commit()

    state = await fence_state(session_id)
    assert state["run_generation"] == 1
    assert state["stop_requested"] is False
    assert state["owner_epoch"] == 1
    assert state["owner_token"] is None
    assert state["lease_until"] is None

    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=now)


async def test_release_then_reclaim_keeps_generation_bumps_epoch(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None
    assert fence.generation == 0 and fence.epoch == 1

    async with session_factory() as db:
        await release(db, fence)
    state = await fence_state(session_id)
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 1
    assert state["owner_token"] is None
    assert state["lease_until"] is None

    again = None
    async with session_factory() as db:
        again = await claim(db, session_id, lease_seconds=60, now=now)
    assert again is not None
    assert again.generation == 0
    assert again.epoch == 2

    async with session_factory() as db:
        await release(db, again)  # the new owner releases the run
    state = await fence_state(session_id)
    assert state["run_generation"] == 0
    assert state["owner_epoch"] == 2
    assert state["owner_token"] is None


async def test_fence_check_guards_message_insert(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    t0 = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=t0)
    assert fence is not None
    expired = t0 + timedelta(seconds=61)

    # Lost fence: the pending message insert is rolled back with the transaction.
    async with session_factory() as db:
        db.add(Message(session_id=session_id, role="user", content="stale"))
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=expired)
        await db.rollback()
    assert await count_messages(session_id) == 0

    # Live fence: the message persists after a successful commit.
    async with session_factory() as db:
        db.add(Message(session_id=session_id, role="user", content="live"))
        await fence_check(db, fence, now=t0)
        await db.commit()
    async with session_factory() as db:
        rows = list(await db.scalars(select(Message).where(Message.session_id == session_id)))
    assert len(rows) == 1
    assert rows[0].role == "user"
    assert rows[0].content == "live"


async def test_release_by_foreign_or_stale_fence_does_not_clear_owner(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None

    # A foreign fence (wrong token) releases nothing while the lease is alive.
    foreign = RunFence(session_id, fence.generation, fence.epoch, "wrong-token")
    async with session_factory() as db:
        await release(db, foreign)
    state = await fence_state(session_id)
    assert state["owner_token"] == fence.token
    assert state["owner_epoch"] == 1
    assert state["lease_until"] is not None

    # A stale fence (previous epoch) does not clear the new owner after a reclaim.
    expired = now + timedelta(seconds=61)
    async with session_factory() as db:
        new_fence = await claim(db, session_id, lease_seconds=60, now=expired)
    assert new_fence is not None and new_fence.epoch == 2
    async with session_factory() as db:
        await release(db, fence)
    state = await fence_state(session_id)
    assert state["owner_token"] == new_fence.token
    assert state["owner_epoch"] == 2


async def test_two_concurrent_claims_only_one_wins(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)

    async def try_claim() -> RunFence | None:
        async with session_factory() as db:
            return await claim(db, session_id, lease_seconds=60, now=now)

    results = await asyncio.gather(try_claim(), try_claim())
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    winner = winners[0]
    state = await fence_state(session_id)
    assert state["owner_token"] == winner.token
    assert state["owner_epoch"] == winner.epoch
    assert state["run_generation"] == 0


async def test_lease_boundary_exactly_at_and_just_after_expiry(tmp_path: Path) -> None:
    """Lease is owned exactly at ``lease_until`` and lost the instant after."""
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    t0 = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=t0)
    assert fence is not None
    lease_until = t0 + timedelta(seconds=60)

    # Exactly at the boundary the fence still owns the run and may renew.
    async with session_factory() as db:
        await fence_check(db, fence, now=lease_until)
    async with session_factory() as db:
        assert await renew(db, fence, lease_seconds=30, now=lease_until) is True
    state = await fence_state(session_id)
    assert state["lease_until"] is not None
    assert state["lease_until"] == lease_until + timedelta(seconds=30)

    renewed_until = lease_until + timedelta(seconds=30)
    # Exactly at the renewed boundary the fence still owns the run.
    async with session_factory() as db:
        await fence_check(db, fence, now=renewed_until)
    # One microsecond past the renewed boundary the fence is lost.
    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=renewed_until + timedelta(microseconds=1))


async def test_claim_with_nonexistent_or_malformed_session_id_returns_none(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    for bad in ("missing-session", "", "   ", "x" * 100):
        async with session_factory() as db:
            fence = await claim(db, bad, lease_seconds=60, now=now)
        assert fence is None

    # A fabricated fence for a nonexistent session renews nothing and cannot write.
    bogus = RunFence("missing-session", 0, 1, "tok")
    async with session_factory() as db:
        assert await renew(db, bogus, lease_seconds=60, now=now) is False
    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, bogus, now=now)


async def test_stop_then_new_generation_resets_stop_and_allows_claim(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value)
    now = datetime.now(UTC)
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=60, now=now)
    assert fence is not None

    async with session_factory() as db:
        assert await stop_generation(db, session_id) is True
        await db.commit()
    async with session_factory() as db:
        await new_generation(db, session_id)
        await db.commit()

    state = await fence_state(session_id)
    assert state["run_generation"] == 2
    assert state["stop_requested"] is False
    assert state["owner_token"] is None

    again = None
    async with session_factory() as db:
        again = await claim(db, session_id, lease_seconds=60, now=now)
    assert again is not None
    assert again.generation == 2
    assert again.epoch == 2


async def test_stop_generation_scoped_matches_current_generation(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value, run_generation=2)
    async with session_factory() as db:
        assert await stop_generation_scoped(db, session_id, 2) is True
        await db.commit()
    state = await fence_state(session_id)
    assert state["run_generation"] == 3
    assert state["stop_requested"] is True
    assert state["owner_token"] is None
    assert state["lease_until"] is None


async def test_stop_generation_scoped_wrong_generation_is_noop(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value, run_generation=2)
    async with session_factory() as db:
        assert await stop_generation_scoped(db, session_id, 1) is False
    state = await fence_state(session_id)
    assert state["run_generation"] == 2
    assert state["stop_requested"] is False
    assert state["owner_token"] is None
    assert state["status"] == SessionStatus.running.value


async def test_stop_generation_scoped_terminal_status_is_noop(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.completed.value, run_generation=2)
    async with session_factory() as db:
        assert await stop_generation_scoped(db, session_id, 2) is False
    state = await fence_state(session_id)
    assert state["run_generation"] == 2
    assert state["stop_requested"] is False
    assert state["status"] == SessionStatus.completed.value


async def test_stop_generation_scoped_repeated_stop_is_noop(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path, status=SessionStatus.running.value, run_generation=5)
    async with session_factory() as db:
        assert await stop_generation_scoped(db, session_id, 5) is True
        await db.commit()
    async with session_factory() as db:
        assert await stop_generation_scoped(db, session_id, 5) is False
    state = await fence_state(session_id)
    assert state["run_generation"] == 6
    assert state["stop_requested"] is True
