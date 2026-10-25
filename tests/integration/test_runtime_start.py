"""The explicit start contract: a claimed generation is scheduled or refused, never lost."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import fakeredis.aioredis
import pytest
from sqlalchemy import update

import agent.runtime as runtime
from agent.config import Settings, get_settings
from agent.database import init_database, session_factory
from agent.models import Project, Session, SessionStatus
from agent.queue import RedisTaskQueue
from agent.runtime import AgentSupervisor
from agent.session_fence import CURRENT_FENCE, new_generation


async def make_session(root: Path, **values: Any) -> str:
    await init_database(recover_running=False)
    async with session_factory() as db:
        tag = uuid.uuid4().hex[:8]
        project = Project(name=f"start-{tag}", root_path=str(root / tag))
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            title="start",
            status=values.pop("status", SessionStatus.pending.value),
            configuration={},
            **values,
        )
        db.add(session)
        await db.commit()
        return session.id


class RecordingSupervisor(AgentSupervisor):
    """A supervisor whose runs only record the call and wait until released."""

    def __init__(self) -> None:
        super().__init__(get_settings())
        self.runs: list[str] = []
        self.release = asyncio.Event()

    async def _run_guarded(self, session_id: str, generation: int | None = None) -> None:
        self.runs.append(session_id)
        await self.release.wait()


async def test_new_generation_starts_while_old_task_finishes(tmp_path: Path) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False, expected_generation=1)
    old_task = supervisor._tasks[session_id]
    async with session_factory() as db:  # the API claims generation 2
        await db.execute(update(Session).where(Session.id == session_id).values(run_generation=2))
        await db.commit()

    assert await supervisor.start(session_id, new_run=False, expected_generation=2)

    new_task = supervisor._tasks[session_id]
    await asyncio.sleep(0)
    assert new_task is not old_task
    assert not old_task.cancelled()  # the fence, not the start call, stops the old task
    assert supervisor.runs == [session_id, session_id]
    supervisor.release.set()
    await asyncio.gather(old_task, new_task)


async def test_same_generation_is_scheduled_once(tmp_path: Path) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, run_generation=4)
    results = await asyncio.gather(
        *(supervisor.start(session_id, new_run=False, expected_generation=4) for _ in range(3))
    )
    await asyncio.sleep(0)
    assert results == [True, True, True]
    assert supervisor.runs == [session_id]
    supervisor.release.set()
    await supervisor._tasks[session_id]


@pytest.mark.parametrize(
    "state",
    [
        {"run_generation": 3},  # a newer run took over
        {"run_generation": 2, "stop_requested": True},  # a Stop is pending
        {"run_generation": 2, "status": SessionStatus.stopped.value},  # terminal
    ],
)
async def test_stale_generation_is_refused(tmp_path: Path, state: dict[str, Any]) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, **state)
    assert await supervisor.start(session_id, new_run=False, expected_generation=2) is False
    assert session_id not in supervisor._tasks
    assert supervisor.runs == []


async def test_call_without_generation_keeps_old_behaviour(tmp_path: Path) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False)
    assert await supervisor.start(session_id, new_run=False) is False  # task still alive
    supervisor.release.set()
    await supervisor._tasks[session_id]


def fake_queue(namespace: str) -> RedisTaskQueue:
    settings = Settings(
        execution_mode="redis",
        database_url="postgresql+asyncpg://agent:secret@db/agent",
        redis_namespace=namespace,
    )
    queue = RedisTaskQueue(settings)
    queue.client = fakeredis.aioredis.FakeRedis(decode_responses=True)  # type: ignore[assignment]
    return queue


async def test_queue_keeps_one_marker_per_generation() -> None:
    queue = fake_queue("start-generations")
    try:
        await queue.ensure_group()
        assert await queue.enqueue("s1", generation=1)
        assert await queue.enqueue("s1", generation=2)  # not a duplicate of generation 1
        assert await queue.enqueue("s1", generation=2) is False  # same generation
        messages = await queue.read("worker-1", count=10, block_ms=1)
        assert [(m.session_id, m.generation) for m in messages] == [("s1", 1), ("s1", 2)]
        await queue.acknowledge(messages[1].message_id, "s1", 2)
        assert await queue.enqueue("s1", generation=2)  # the marker of generation 2 is gone
        assert await queue.enqueue("s1", generation=1) is False  # generation 1 still queued
    finally:
        await queue.close()


async def test_failed_enqueue_leaves_no_marker() -> None:
    # The stream key has the wrong type, so XADD fails inside the script: no marker may
    # remain, and a later producer must not get a false "already queued".
    queue = fake_queue("start-failure")
    try:
        await queue.client.set(queue.stream, "not a stream")
        # Real Redis answers WRONGTYPE; fakeredis raises its own error type here.
        with pytest.raises(Exception):  # noqa: B017
            await queue.enqueue("s1", generation=1)
        assert await queue.client.exists(queue._dedupe_key("s1", 1)) == 0
        await queue.client.delete(queue.stream)
        await queue.ensure_group()
        assert await queue.enqueue("s1", generation=1) is True
    finally:
        await queue.close()


async def test_stale_job_cannot_claim_a_newer_generation(tmp_path: Path) -> None:
    # The real guard: a job scheduled for generation 1 waits for a slot; meanwhile the
    # session moves to generation 2. The claim must refuse, and the run must not start.
    supervisor = AgentSupervisor(get_settings())
    executed: list[int] = []

    async def fake_run(session_id: str) -> None:
        fence = CURRENT_FENCE.get()
        executed.append(fence.generation if fence else -1)

    supervisor._run = fake_run  # type: ignore[method-assign]
    session_id = await make_session(tmp_path, run_generation=1)
    slots = supervisor.settings.max_parallel_sessions
    for _ in range(slots):  # hold every slot: the job must wait
        await supervisor._semaphore.acquire()
    try:
        assert await supervisor.start(session_id, new_run=False, expected_generation=1)
        async with session_factory() as db:
            await db.execute(
                update(Session).where(Session.id == session_id).values(run_generation=2)
            )
            await db.commit()
        await asyncio.sleep(0.05)
        assert executed == [] and not supervisor._tasks[session_id].done()  # still waiting
    finally:
        for _ in range(slots):
            supervisor._semaphore.release()
    await supervisor._tasks[session_id]
    assert executed == []
    assert await supervisor.start(session_id, new_run=False, expected_generation=2)
    await supervisor._tasks[session_id]
    assert executed == [2]


async def test_shutdown_and_stop_cancel_finishing_tasks(tmp_path: Path) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False, expected_generation=1)
    old_task = supervisor._tasks[session_id]
    async with session_factory() as db:
        await db.execute(update(Session).where(Session.id == session_id).values(run_generation=2))
        await db.commit()
    assert await supervisor.start(session_id, new_run=False, expected_generation=2)
    new_task = supervisor._tasks[session_id]
    await asyncio.sleep(0)
    await supervisor.stop(session_id)
    assert old_task.cancelled() and new_task.cancelled()

    other = RecordingSupervisor()
    second = await make_session(tmp_path, run_generation=1)
    assert await other.start(second, new_run=False, expected_generation=1)
    first_task = other._tasks[second]
    async with session_factory() as db:
        await db.execute(update(Session).where(Session.id == second).values(run_generation=2))
        await db.commit()
    assert await other.start(second, new_run=False, expected_generation=2)
    await asyncio.sleep(0)
    await other.shutdown()
    assert first_task.done() and other._tasks[second].done()


async def test_expected_generation_with_new_run_is_rejected(tmp_path: Path) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    with pytest.raises(ValueError, match="new_run=False"):
        await supervisor.start(session_id, expected_generation=1)
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None and session.run_generation == 1
    assert session_id not in supervisor._tasks


async def test_stop_scoped_does_not_affect_newer_generation(tmp_path: Path) -> None:
    supervisor = RecordingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False, expected_generation=1)
    old_task = supervisor._tasks[session_id]
    async with session_factory() as db:
        await db.execute(update(Session).where(Session.id == session_id).values(run_generation=2))
        await db.commit()
    assert await supervisor.start(session_id, new_run=False, expected_generation=2)
    new_task = supervisor._tasks[session_id]
    await asyncio.sleep(0)

    # A stale stop of generation 1 must not affect generation 2.
    assert await supervisor.stop_scoped(session_id, 1) is False
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.run_generation == 2
        assert session.stop_requested is False
    assert not new_task.cancelled()

    # A stop of the current generation cancels its task.
    assert await supervisor.stop_scoped(session_id, 2) is True
    assert new_task.cancelled()

    supervisor.release.set()
    await asyncio.gather(old_task, return_exceptions=True)


# --- scoped Stop after the commit: events fail, the caller is cancelled, a newer run starts


class WaitingSupervisor(AgentSupervisor):
    """Hold each scheduled task at an explicit barrier and report its generation."""

    def __init__(self) -> None:
        super().__init__(get_settings())
        self.entered: asyncio.Queue[int | None] = asyncio.Queue()
        self.release = asyncio.Event()

    async def _run(self, session_id: str) -> None:
        fence = CURRENT_FENCE.get()
        assert fence is not None
        await self.entered.put(fence.generation)
        await self.release.wait()


async def assert_stopped(session_id: str) -> None:
    async with session_factory() as db:
        stored = await db.get(Session, session_id)
        assert stored is not None
        assert stored.status == SessionStatus.stopped.value
        assert stored.stop_requested is True
        assert stored.run_generation == 2


async def test_event_failure_after_commit_still_cancels_expected_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = WaitingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False, expected_generation=1)
    assert await asyncio.wait_for(supervisor.entered.get(), 2) == 1
    task = supervisor._tasks[session_id]

    async def broken_event(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic event failure")

    monkeypatch.setattr(runtime, "publish", broken_event)
    try:
        # A time limit: without the immediate cancel only the lease loss (60 s) ends the task.
        with pytest.raises(RuntimeError, match="synthetic event failure"):
            async with asyncio.timeout(5):
                await supervisor.stop_scoped(session_id, 1)
        await assert_stopped(session_id)
        assert task.cancelled()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancelled_stop_after_commit_still_cancels_expected_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = WaitingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False, expected_generation=1)
    assert await asyncio.wait_for(supervisor.entered.get(), 2) == 1
    task = supervisor._tasks[session_id]
    in_event = asyncio.Event()
    release_event = asyncio.Event()

    async def blocked_event(*args: object, **kwargs: object) -> None:
        in_event.set()
        await release_event.wait()

    monkeypatch.setattr(runtime, "publish", blocked_event)
    stop_task = asyncio.create_task(supervisor.stop_scoped(session_id, 1))
    try:
        await asyncio.wait_for(in_event.wait(), 2)
        await assert_stopped(session_id)
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5):
                await stop_task
        assert task.cancelled()
    finally:
        release_event.set()
        stop_task.cancel()
        task.cancel()
        await asyncio.gather(stop_task, task, return_exceptions=True)


async def test_replaced_registry_after_commit_cancels_only_old_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = WaitingSupervisor()
    session_id = await make_session(tmp_path, run_generation=1)
    assert await supervisor.start(session_id, new_run=False, expected_generation=1)
    assert await asyncio.wait_for(supervisor.entered.get(), 2) == 1
    old_task = supervisor._tasks[session_id]
    in_event = asyncio.Event()
    release_event = asyncio.Event()

    async def blocked_event(*args: object, **kwargs: object) -> None:
        in_event.set()
        await release_event.wait()

    monkeypatch.setattr(runtime, "publish", blocked_event)
    stop_task = asyncio.create_task(supervisor.stop_scoped(session_id, 1))
    tasks = [old_task, stop_task]
    try:
        await asyncio.wait_for(in_event.wait(), 2)
        await assert_stopped(session_id)
        async with session_factory() as db:
            await new_generation(db, session_id)
            await db.execute(
                update(Session).where(Session.id == session_id).values(status="pending")
            )
            await db.commit()
        assert await supervisor.start(session_id, new_run=False, expected_generation=3)
        assert await asyncio.wait_for(supervisor.entered.get(), 2) == 3
        new_task = supervisor._tasks[session_id]
        tasks.append(new_task)
        assert new_task is not old_task
        release_event.set()
        async with asyncio.timeout(5):
            assert await stop_task is True
        assert not new_task.done()
        async with session_factory() as db:
            stored = await db.get(Session, session_id)
            assert stored is not None
            assert stored.run_generation == 3
            assert stored.stop_requested is False
            assert stored.status == "pending"
        assert old_task.cancelled()
    finally:
        release_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
