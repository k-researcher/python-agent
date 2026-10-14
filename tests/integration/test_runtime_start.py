"""The explicit start contract: a claimed generation is scheduled or refused, never lost."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import fakeredis.aioredis
import pytest
from sqlalchemy import update

from agent.config import Settings, get_settings
from agent.database import init_database, session_factory
from agent.models import Project, Session, SessionStatus
from agent.queue import RedisTaskQueue
from agent.runtime import AgentSupervisor
from agent.session_fence import CURRENT_FENCE


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
