from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from agent import api
from agent.database import init_database, session_factory
from agent.models import Event, Message, Project, Session
from agent.schemas import MessageCreate


@pytest.fixture
async def history(tmp_path: Path) -> AsyncIterator[tuple[str, int]]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name="submission", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            title="submission",
            status="completed",
            run_generation=7,
            error="old error",
            stop_requested=True,
        )
        db.add(session)
        await db.flush()
        root = Message(session_id=session.id, role="system", content="root")
        db.add(root)
        await db.flush()
        user = Message(session_id=session.id, parent_id=root.id, role="user", content="task")
        db.add(user)
        await db.commit()
        yield session.id, user.id


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    calls: list[tuple[str, bool]] = []

    async def start(
        session_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        assert expected_generation is not None
        async with session_factory() as db:
            session = await db.get(Session, session_id)
            assert session is not None and session.run_generation == expected_generation
        calls.append((session_id, new_run))
        return True

    monkeypatch.setattr(api.supervisor, "start", start)
    return calls


@pytest.mark.parametrize("status", ["completed", "stopped", "error", "interrupted"])
async def test_idle_submission_claims_generation_and_appends_once(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
    status: str,
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(update(Session).where(Session.id == session_id).values(status=status))
        await db.commit()
        message = await api.add_message(session_id, MessageCreate(content="next"), db)
        assert message.parent_id == head_id
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.status == "pending"
        assert session.run_generation == 8
        assert session.stop_requested is False
        assert session.error is None
        assert session.owner_token is None and session.lease_until is None
    assert starts == [(session_id, False)]


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
async def test_busy_submission_does_not_change_state_or_history(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
    status: str,
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(update(Session).where(Session.id == session_id).values(status=status))
        await db.commit()
        with pytest.raises(HTTPException) as caught:
            await api.add_message(session_id, MessageCreate(content="late"), db)
        assert caught.value.status_code == 409
        await db.rollback()
        session = await db.get(Session, session_id)
        assert session is not None and session.status == status and session.run_generation == 7
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == []


@pytest.mark.parametrize("lease_offset", [60, None])
@pytest.mark.parametrize("operation", ["send", "run"])
async def test_owner_with_live_or_unknown_lease_blocks_submission(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
    lease_offset: int | None,
    operation: str,
) -> None:
    session_id, _ = history
    lease = datetime.now(UTC) + timedelta(seconds=lease_offset) if lease_offset else None
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                owner_token="owner",
                lease_until=lease,
            )
        )
        await db.commit()
        with pytest.raises(HTTPException) as caught:
            if operation == "send":
                await api.add_message(session_id, MessageCreate(content="late"), db)
            else:
                await api.start_session(session_id, db)
        assert caught.value.status_code == 409
        await db.rollback()
    assert starts == []


async def test_expired_owner_is_invalidated_on_submission(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, _ = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                owner_token="expired",
                lease_until=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
        await db.commit()
        await api.add_message(session_id, MessageCreate(content="next"), db)
        session = await db.get(Session, session_id, populate_existing=True)
        assert session is not None
        assert session.owner_token is None and session.run_generation == 8
    assert starts == [(session_id, False)]


async def test_stale_idle_identity_cannot_overwrite_running(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, head_id = history
    async with session_factory() as stale:
        cached = await stale.get(Session, session_id)
        assert cached is not None and cached.status == "completed"
        async with session_factory() as other:
            await other.execute(
                update(Session)
                .where(Session.id == session_id)
                .values(
                    status="running",
                    owner_token="other",
                    stop_requested=False,
                    lease_until=datetime.now(UTC) + timedelta(minutes=1),
                )
            )
            await other.commit()
        with pytest.raises(HTTPException) as caught:
            await api.add_message(session_id, MessageCreate(content="late"), stale)
        assert caught.value.status_code == 409
        await stale.rollback()
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert (
            session is not None and session.status == "running" and session.owner_token == "other"
        )
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == []


async def test_second_submission_cannot_join_pending_run(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, _ = history
    async with session_factory() as first:
        await api.add_message(session_id, MessageCreate(content="first"), first)
    async with session_factory() as second:
        with pytest.raises(HTTPException) as caught:
            await api.add_message(session_id, MessageCreate(content="second"), second)
        assert caught.value.status_code == 409
        await second.rollback()
    assert starts == [(session_id, False)]


async def test_bad_reasoning_rolls_back_claim(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        with pytest.raises(HTTPException) as caught:
            await api.add_message(
                session_id, MessageCreate(content="next", reasoning_effort="wrong"), db
            )
        assert caught.value.status_code == 422
        await db.rollback()
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.status == "completed" and session.run_generation == 7
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == []


async def test_missing_session_is_404(starts: list[tuple[str, bool]]) -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        with pytest.raises(HTTPException) as caught:
            await api.add_message("missing", MessageCreate(content="next"), db)
        assert caught.value.status_code == 404
    assert starts == []


async def test_run_cannot_claim_a_run_prepared_by_submission(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, _ = history
    async with session_factory() as first:
        await api.add_message(session_id, MessageCreate(content="first"), first)
    async with session_factory() as second:
        with pytest.raises(HTTPException) as caught:
            await api.start_session(session_id, second)
        assert caught.value.status_code == 409
        await second.rollback()
    assert starts == [(session_id, False)]


async def test_stale_run_request_cannot_overwrite_running(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, _ = history
    async with session_factory() as stale:
        cached = await stale.get(Session, session_id)
        assert cached is not None and cached.status == "completed"
        async with session_factory() as other:
            await other.execute(
                update(Session)
                .where(Session.id == session_id)
                .values(
                    status="running",
                    owner_token="other",
                    stop_requested=False,
                    lease_until=datetime.now(UTC) + timedelta(minutes=1),
                )
            )
            await other.commit()
        with pytest.raises(HTTPException) as caught:
            await api.start_session(session_id, stale)
        assert caught.value.status_code == 409
        await stale.rollback()
    assert starts == []


async def test_initial_pending_session_can_start_once(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, _ = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                status="pending",
                run_generation=0,
                stop_requested=False,
                error=None,
            )
        )
        await db.commit()
        assert await api.start_session(session_id, db) == {"started": True}
        session = await db.get(Session, session_id, populate_existing=True)
        assert session is not None and session.run_generation == 1
    assert starts == [(session_id, False)]


async def test_two_concurrent_submissions_have_one_winner(
    history: tuple[str, int],
    starts: list[tuple[str, bool]],
) -> None:
    session_id, _ = history

    async def submit(content: str) -> int:
        async with session_factory() as db:
            try:
                await api.add_message(session_id, MessageCreate(content=content), db)
                return 201
            except HTTPException as exc:
                await db.rollback()
                return exc.status_code

    outcomes = await asyncio.gather(submit("first"), submit("second"))
    assert sorted(outcomes) == [201, 409]
    assert starts == [(session_id, False)]


async def test_stop_after_commit_is_not_undone_by_delayed_start(
    history: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _ = history
    executed: list[str] = []

    original_start = api.supervisor.start

    async def stop_before_start(
        target_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        await api.supervisor.stop(target_id)
        return await original_start(
            target_id, new_run=new_run, expected_generation=expected_generation
        )

    async def execute(target_id: str) -> None:
        executed.append(target_id)

    monkeypatch.setattr(api.supervisor, "start", stop_before_start)
    monkeypatch.setattr(api.supervisor, "_run", execute)
    async with session_factory() as db:
        await api.add_message(session_id, MessageCreate(content="first"), db)
    task = api.supervisor._tasks.get(session_id)
    if task is not None:
        await asyncio.wait_for(task, timeout=3)
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.status == "stopped" and session.run_generation == 9
        assert session.stop_requested is True
    assert executed == []


@pytest.mark.parametrize("status", ["completed", "stopped", "error", "interrupted"])
async def test_idle_run_claims_once_without_appending(
    history: tuple[str, int], starts: list[tuple[str, bool]], status: str
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(update(Session).where(Session.id == session_id).values(status=status))
        await db.commit()
        assert await api.start_session(session_id, db) == {"started": True}
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.status == "pending" and session.run_generation == 8
        assert session.stop_requested is False and session.error is None
        assert session.owner_token is None and session.lease_until is None
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == [(session_id, False)]


async def test_run_rejects_awaiting_confirmation_without_changes(
    history: tuple[str, int], starts: list[tuple[str, bool]]
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(
            update(Session).where(Session.id == session_id).values(status="awaiting_confirmation")
        )
        await db.commit()
        with pytest.raises(HTTPException) as caught:
            await api.start_session(session_id, db)
        assert caught.value.status_code == 409
        await db.rollback()
        session = await db.get(Session, session_id)
        assert session is not None and session.status == "awaiting_confirmation"
        assert session.run_generation == 7 and session.stop_requested is True
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == []


async def test_initial_pending_submission_is_allowed_once(
    history: tuple[str, int], starts: list[tuple[str, bool]]
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status="pending", run_generation=0)
        )
        await db.commit()
        message = await api.add_message(session_id, MessageCreate(content="first"), db)
        assert message.parent_id == head_id
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.run_generation == 1 and session.status == "pending"
        with pytest.raises(HTTPException) as caught:
            await api.add_message(session_id, MessageCreate(content="duplicate"), db)
        assert caught.value.status_code == 409
        await db.rollback()
        assert (
            await db.scalar(select(func.count(Message.id)).where(Message.session_id == session_id))
            == 3
        )
    assert starts == [(session_id, False)]


async def test_repeated_run_does_not_claim_pending_generation(
    history: tuple[str, int], starts: list[tuple[str, bool]]
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status="pending", run_generation=0)
        )
        await db.commit()
        await api.start_session(session_id, db)
    async with session_factory() as db:
        with pytest.raises(HTTPException) as caught:
            await api.start_session(session_id, db)
        assert caught.value.status_code == 409
        await db.rollback()
        session = await db.get(Session, session_id)
        assert session is not None and session.run_generation == 1 and session.status == "pending"
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == [(session_id, False)]


async def test_concurrent_initial_runs_have_one_winner(
    history: tuple[str, int], starts: list[tuple[str, bool]]
) -> None:
    session_id, _ = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status="pending", run_generation=0)
        )
        await db.commit()

    async def start() -> int:
        async with session_factory() as db:
            try:
                await api.start_session(session_id, db)
                return 202
            except HTTPException as exc:
                await db.rollback()
                return exc.status_code

    assert sorted(await asyncio.gather(start(), start())) == [202, 409]
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.run_generation == 1
    assert starts == [(session_id, False)]


@pytest.mark.parametrize("operation", ["send", "run"])
async def test_expired_running_session_requires_explicit_recovery(
    history: tuple[str, int], starts: list[tuple[str, bool]], operation: str
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                status="running",
                owner_token="expired",
                lease_until=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
        await db.commit()
        with pytest.raises(HTTPException) as caught:
            if operation == "send":
                await api.add_message(session_id, MessageCreate(content="late"), db)
            else:
                await api.start_session(session_id, db)
        assert caught.value.status_code == 409
        await db.rollback()
        session = await db.get(Session, session_id)
        assert session is not None and session.status == "running" and session.run_generation == 7
        assert session.owner_token == "expired"
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
    assert starts == []


async def test_publication_failure_does_not_lose_dispatch_or_saved_event(
    history: tuple[str, int], starts: list[tuple[str, bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, _ = history

    async def publish(event: object) -> None:
        assert starts == [(session_id, False)]
        raise RuntimeError("publication failed")

    monkeypatch.setattr(api.broker, "broadcast", publish)
    async with session_factory() as db:
        message = await api.add_message(session_id, MessageCreate(content="next"), db)
    async with session_factory() as db:
        event = await db.scalar(
            select(Event).where(Event.session_id == session_id, Event.type == "message.created")
        )
        assert event is not None and event.payload == {
            "role": "user",
            "message_id": message.id,
            "generation": 8,
        }
        assert await db.get(Message, message.id) is not None


async def test_event_insert_failure_rolls_back_claim_and_message(
    history: tuple[str, int], starts: list[tuple[str, bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, head_id = history
    async with session_factory() as db:
        original_flush = db.flush

        async def flush(objects: object = None) -> None:
            if any(isinstance(row, Event) for row in db.new):
                raise RuntimeError("event insert failed")
            await original_flush()

        monkeypatch.setattr(db, "flush", flush)
        with pytest.raises(RuntimeError, match="event insert failed"):
            await api.add_message(session_id, MessageCreate(content="next"), db)
        await db.rollback()
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.status == "completed" and session.run_generation == 7
        assert (
            await db.scalar(select(func.max(Message.id)).where(Message.session_id == session_id))
            == head_id
        )
        assert (
            await db.scalar(select(func.count(Event.id)).where(Event.session_id == session_id)) == 0
        )
    assert starts == []


@pytest.mark.parametrize("operation", ["send", "run"])
@pytest.mark.parametrize("race", ["none", "stop", "newer", "owner"])
async def test_dispatch_error_does_not_leave_unowned_pending_or_overwrite_race(
    history: tuple[str, int], monkeypatch: pytest.MonkeyPatch, operation: str, race: str
) -> None:
    session_id, _ = history

    async def fail(
        target_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        assert target_id == session_id and new_run is False
        assert expected_generation == 8
        if race != "none":
            async with session_factory() as other:
                changes: dict[str, object]
                if race == "stop":
                    changes = {"status": "stopped", "run_generation": 9, "stop_requested": True}
                elif race == "newer":
                    changes = {"status": "pending", "run_generation": 9}
                else:
                    changes = {"status": "running", "owner_token": "current"}
                await other.execute(
                    update(Session).where(Session.id == session_id).values(**changes)
                )
                await other.commit()
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(api.supervisor, "start", fail)
    async with session_factory() as db:
        with pytest.raises(HTTPException) as caught:
            if operation == "send":
                await api.add_message(session_id, MessageCreate(content="next"), db)
            else:
                await api.start_session(session_id, db)
        assert caught.value.status_code == 503
        assert "retry Start, not Send" in caught.value.detail
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        expected = {"none": "error", "stop": "stopped", "newer": "pending", "owner": "running"}
        assert session.status == expected[race]
        assert session.run_generation == (9 if race in {"stop", "newer"} else 8)
        assert session.stop_requested is (race == "stop")
        assert session.owner_token == ("current" if race == "owner" else None)
        assert await db.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        ) == (3 if operation == "send" else 2)
        if operation == "send":
            event = await db.scalar(
                select(Event).where(Event.session_id == session_id, Event.type == "message.created")
            )
            assert event is not None
        if race == "none":
            assert session.error == "The run could not be scheduled; start it again"


@pytest.mark.parametrize("operation", ["send", "run"])
@pytest.mark.parametrize("race", ["none", "stop", "newer", "owner"])
async def test_cancelled_dispatch_preserves_history_and_does_not_overwrite_race(
    history: tuple[str, int], monkeypatch: pytest.MonkeyPatch, operation: str, race: str
) -> None:
    session_id, _ = history
    entered = asyncio.Event()

    async def wait_for_cancel(
        target_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        assert target_id == session_id and new_run is False
        assert expected_generation == 8
        if race != "none":
            async with session_factory() as other:
                changes: dict[str, object]
                if race == "stop":
                    changes = {"status": "stopped", "run_generation": 9, "stop_requested": True}
                elif race == "newer":
                    changes = {"status": "pending", "run_generation": 9}
                else:
                    changes = {"status": "running", "owner_token": "current"}
                await other.execute(
                    update(Session).where(Session.id == session_id).values(**changes)
                )
                await other.commit()
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("The dispatch must be cancelled")

    monkeypatch.setattr(api.supervisor, "start", wait_for_cancel)

    async def submit() -> None:
        async with session_factory() as db:
            if operation == "send":
                await api.add_message(session_id, MessageCreate(content="next"), db)
            else:
                await api.start_session(session_id, db)

    request = asyncio.create_task(submit())
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    finally:
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)

    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        expected = {"none": "error", "stop": "stopped", "newer": "pending", "owner": "running"}
        assert session.status == expected[race]
        assert session.run_generation == (9 if race in {"stop", "newer"} else 8)
        assert session.stop_requested is (race == "stop")
        assert session.owner_token == ("current" if race == "owner" else None)
        assert await db.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        ) == (3 if operation == "send" else 2)
        if operation == "send":
            event = await db.scalar(
                select(Event).where(Event.session_id == session_id, Event.type == "message.created")
            )
            assert event is not None
        if race == "none":
            assert session.error == "The run could not be scheduled; start it again"


@pytest.mark.parametrize("operation", ["send", "run"])
async def test_dispatch_failure_can_be_retried_by_run_without_duplicate_message(
    history: tuple[str, int], monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    session_id, _ = history
    calls = 0

    async def dispatch(
        target_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        nonlocal calls
        assert target_id == session_id and new_run is False
        calls += 1
        assert expected_generation == 7 + calls
        if calls == 1:
            raise RuntimeError("dispatch failed")
        return True

    monkeypatch.setattr(api.supervisor, "start", dispatch)
    async with session_factory() as db:
        with pytest.raises(HTTPException) as caught:
            if operation == "send":
                await api.add_message(session_id, MessageCreate(content="next"), db)
            else:
                await api.start_session(session_id, db)
        assert caught.value.status_code == 503
    async with session_factory() as db:
        assert await api.start_session(session_id, db) == {"started": True}
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.run_generation == 9 and session.status == "pending"
        assert session.error is None
        assert await db.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        ) == (3 if operation == "send" else 2)
    assert calls == 2


@pytest.mark.parametrize("second_operation", ["send", "run"])
async def test_initial_send_race_has_one_generation_and_no_duplicate_message(
    history: tuple[str, int], starts: list[tuple[str, bool]], second_operation: str
) -> None:
    session_id, _ = history
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status="pending", run_generation=0)
        )
        await db.commit()

    async def submit(operation: str) -> tuple[str, int]:
        async with session_factory() as db:
            try:
                if operation == "send":
                    await api.add_message(session_id, MessageCreate(content="first"), db)
                    return operation, 201
                await api.start_session(session_id, db)
                return operation, 202
            except HTTPException as exc:
                await db.rollback()
                return operation, exc.status_code

    outcomes = await asyncio.gather(submit("send"), submit(second_operation))
    winners = [operation for operation, code in outcomes if code != 409]
    assert len(winners) == 1
    assert all(code in {201, 202, 409} for _, code in outcomes)
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.status == "pending" and session.run_generation == 1
        assert await db.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        ) == (3 if winners[0] == "send" else 2)
    assert starts == [(session_id, False)]


async def test_dispatch_observes_message_event_and_generation_committed_together(
    history: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, _ = history

    async def dispatch(
        target_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        assert target_id == session_id and new_run is False
        assert expected_generation == 8
        async with session_factory() as other:
            session = await other.get(Session, session_id)
            assert (
                session is not None and session.run_generation == 8 and session.status == "pending"
            )
            event = await other.scalar(
                select(Event).where(Event.session_id == session_id, Event.type == "message.created")
            )
            assert event is not None
            message = await other.get(Message, event.payload["message_id"])
            assert message is not None and message.content == "next"
        return True

    monkeypatch.setattr(api.supervisor, "start", dispatch)
    async with session_factory() as db:
        await api.add_message(session_id, MessageCreate(content="next"), db)


async def test_missing_run_session_is_404(starts: list[tuple[str, bool]]) -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        with pytest.raises(HTTPException) as caught:
            await api.start_session("missing", db)
        assert caught.value.status_code == 404
    assert starts == []


@pytest.mark.parametrize("operation", ["send", "run"])
@pytest.mark.parametrize("race", ["none", "stop", "newer"])
async def test_dispatch_refusal_cannot_silently_strand_current_generation(
    history: tuple[str, int], monkeypatch: pytest.MonkeyPatch, operation: str, race: str
) -> None:
    session_id, _ = history

    async def refuse(
        target_id: str, *, new_run: bool = True, expected_generation: int | None = None
    ) -> bool:
        assert target_id == session_id and new_run is False and expected_generation == 8
        if race != "none":
            async with session_factory() as other:
                await other.execute(
                    update(Session)
                    .where(Session.id == session_id)
                    .values(
                        run_generation=9,
                        status="stopped" if race == "stop" else "pending",
                        stop_requested=race == "stop",
                    )
                )
                await other.commit()
        return False

    monkeypatch.setattr(api.supervisor, "start", refuse)
    async with session_factory() as db:
        if race == "none":
            with pytest.raises(HTTPException) as caught:
                if operation == "send":
                    await api.add_message(session_id, MessageCreate(content="next"), db)
                else:
                    await api.start_session(session_id, db)
            assert caught.value.status_code == 503
        elif operation == "send":
            await api.add_message(session_id, MessageCreate(content="next"), db)
        else:
            assert await api.start_session(session_id, db) == {"started": False}
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.status == {"none": "error", "stop": "stopped", "newer": "pending"}[race]
        assert session.run_generation == (8 if race == "none" else 9)
        assert await db.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        ) == (3 if operation == "send" else 2)
