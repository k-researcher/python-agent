"""Native decisions belong to one request and the still-owned turn, across processes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from agent.database import session_factory
from agent.engine import ApprovalEvent, DoneEvent
from agent.events import broker
from agent.models import Approval, Message, Session, ToolCall
from agent.runtime import AgentSupervisor
from tests.integration.test_engine_runner import FakeEngine, rows, setup


class Clock:
    def __init__(self):
        self.time = 0.0
        self.waiting = asyncio.Event()
        self.tick = asyncio.Event()

    def now(self):
        return self.time

    async def until(self, deadline):
        self.waiting.set()
        while self.time < deadline:
            tick = self.tick
            await tick.wait()

    def advance(self, seconds):
        self.time += seconds
        self.tick.set()
        self.tick = asyncio.Event()


def request(request_id="one", scope="request"):
    return ApprovalEvent(
        request_id=request_id,
        item_id=f"item-{request_id}",
        action="write",
        parameters={"path": "a.txt", "content": "original"},
        scope=scope,
    )


async def required(queue, task):
    while True:
        reader = asyncio.create_task(queue.get())
        try:
            done, _ = await asyncio.wait({reader, task}, return_when=asyncio.FIRST_COMPLETED)
            assert reader in done, "Engine ended before publishing approval.required"
            event = reader.result()
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if event["type"] == "approval.required":
            return event["payload"]


async def approval_rows(sid):
    async with session_factory() as db:
        return list(
            await db.scalars(select(Approval).join(ToolCall).where(ToolCall.session_id == sid))
        )


@pytest.mark.parametrize("decision,wire", [("approve", "allow"), ("reject", "deny")])
async def test_decision_cross_process_keeps_fence_and_exact_request(
    tmp_path, monkeypatch, decision, wire
):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [request(), request("two", "turn"), DoneEvent(status="completed", reason=None)]
        )
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        api = AgentSupervisor(worker.settings)  # no engine instance or task in this process
        async with broker.subscribe() as queue:
            await worker.start(sid, new_run=False, expected_generation=1)
            task = worker._tasks[sid]
            try:
                first = await required(queue, task)
                assert fake.answers == [] and not task.done()
                session = (await rows(sid))[0]
                assert session.owner_token and session.status == "awaiting_confirmation"
                assert first["generation"] == 1 and first["executor"] == "engine"
                assert first["scope"] == "request" and first["request_id"] == "one"
                await api.resolve_approval(
                    first["approval_id"],
                    decision,
                    None,
                    actor_user_id="operator",
                    generation=1,
                    executor="engine",
                )
                with pytest.raises(ValueError):
                    await api.resolve_approval(
                        first["approval_id"],
                        decision,
                        None,
                        actor_user_id="operator",
                        generation=1,
                        executor="engine",
                    )
                second = await required(queue, task)
                assert second["scope"] == "turn" and second["request_id"] == "two"
                assert fake.answers == [("one", wire)]
                with pytest.raises(PermissionError):
                    await api.resolve_approval(
                        second["approval_id"],
                        "approve",
                        None,
                        actor_user_id="foreign",
                        generation=1,
                        executor="engine",
                    )
                for generation, executor in [(0, "engine"), (1, "python"), (None, None)]:
                    with pytest.raises(ValueError):
                        await api.resolve_approval(
                            second["approval_id"],
                            "approve",
                            None,
                            actor_user_id="operator",
                            generation=generation,
                            executor=executor,
                        )
                await api.resolve_approval(
                    second["approval_id"],
                    "reject",
                    None,
                    actor_user_id="operator",
                    generation=1,
                    executor="engine",
                )
                await task
            finally:
                await worker.shutdown()
        assert fake.answers == [("one", wire), ("two", "deny")]
        assert len(fake.sent) == 1 and api._tasks == {}
        session, _, turns, _ = await rows(sid)
        assert session.run_generation == 1 and session.status == "completed" and len(turns) == 1
        approvals = await approval_rows(sid)
        assert len(approvals) == 2 and all(a.delivered_at for a in approvals)
        async with session_factory() as db:
            calls = list(await db.scalars(select(ToolCall).where(ToolCall.session_id == sid)))
            assert [c.native_request_id for c in calls] == ["one", "two"]
            assert all(
                c.executor == "engine" and c.arguments == request().parameters for c in calls
            )


@pytest.mark.parametrize("cause", ["ttl", "stop", "lease", "shutdown", "dead"])
async def test_wait_ends_with_deny_and_closed_record(tmp_path, monkeypatch, cause):
    async with asyncio.timeout(4):
        from agent.engine_runner import EngineRunner

        clock = Clock()
        original = EngineRunner.__init__

        def init(self, *args, **kwargs):
            original(self, *args, **kwargs, clock=clock)

        monkeypatch.setattr(EngineRunner, "__init__", init)
        fake = FakeEngine([request(), DoneEvent(status="completed", reason=None)])
        worker, sid = await setup(tmp_path, monkeypatch, fake, timeouts={"approval": 120})
        async with broker.subscribe() as queue:
            await worker.start(sid, new_run=False, expected_generation=1)
            task = worker._tasks[sid]
            try:
                event = await required(queue, task)
                await clock.waiting.wait()
                if cause in {"ttl", "dead"}:
                    if cause == "dead":

                        async def dead(*args):
                            raise BrokenPipeError("closed")

                        fake.answer = dead
                    clock.advance(120)
                    await task
                elif cause == "stop":
                    await worker.stop(sid)
                elif cause == "shutdown":
                    await worker.shutdown()
                else:
                    async with session_factory() as db:
                        await db.execute(
                            update(Session)
                            .where(Session.id == sid)
                            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
                        )
                        await db.commit()
                    clock.advance(1)
                    await asyncio.gather(task, return_exceptions=True)
                record = (await approval_rows(sid))[0]
                assert record.status == ("interrupted" if cause == "dead" else "cancelled")
                assert record.resolved_at is not None
                if cause != "dead":
                    assert fake.answers == [("one", "deny")]
                assert fake.closed.is_set() and fake.interrupts
                with pytest.raises(ValueError):
                    await worker.resolve_approval(
                        event["approval_id"],
                        "approve",
                        None,
                        actor_user_id="operator",
                        generation=1,
                        executor="engine",
                    )
                assert not any(d == "allow" for _, d in fake.answers)
            finally:
                await worker.shutdown()


async def test_user_input_is_empty_answer_and_interruption(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        from agent.engine import UserInputEvent

        fake = FakeEngine([UserInputEvent(request_id="question", item_id="q")])
        answers = []

        async def empty(request_id):
            answers.append(request_id)

        fake.answer_user_input = empty
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        await worker.run_job(sid, 1)
        assert answers == ["question"] and fake.answers == []
        session, _, _, outcomes = await rows(sid)
        assert session.status == "interrupted"
        assert "ввод" in outcomes[0].reason and "поддерж" in outcomes[0].reason
        assert await approval_rows(sid) == []


async def test_duplicate_native_request_never_reuses_a_decision(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine([request(), request(), DoneEvent(status="completed", reason=None)])
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        async with broker.subscribe() as queue:
            await worker.start(sid, new_run=False, expected_generation=1)
            try:
                event = await required(queue, worker._tasks[sid])
                await worker.resolve_approval(
                    event["approval_id"],
                    "approve",
                    None,
                    actor_user_id="operator",
                    generation=1,
                    executor="engine",
                )
                await worker._tasks[sid]
            finally:
                await worker.shutdown()
        assert fake.answers.count(("one", "allow")) == 1
        assert (await rows(sid))[0].status == "error"
        assert len(await approval_rows(sid)) == 1


async def test_recovery_closes_undelivered_approval_after_worker_loss(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        from agent.engine_journal import recover_engines
        from agent.models import EngineTurn

        fake = FakeEngine()
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        async with session_factory() as db:
            session = await db.get(Session, sid)
            session.status = "awaiting_confirmation"
            session.lease_until = datetime.now(UTC) - timedelta(seconds=1)
            turn = EngineTurn(
                session_id=sid,
                generation=1,
                input_message_id=await db.scalar(
                    select(Message.id).where(Message.session_id == sid, Message.role == "user")
                ),
                state="dispatched",
            )
            db.add(turn)
            call = ToolCall(
                id="orphan-call",
                session_id=sid,
                name="write",
                arguments={},
                risk_level="observed",
                executor="engine",
                native_request_id="orphan",
                status="awaiting_confirmation",
            )
            db.add(call)
            await db.flush()
            db.add(Approval(tool_call_id=call.id, executor="engine", generation=1))
            await db.commit()
        async with session_factory() as db:
            await recover_engines(db)
            await db.commit()
        record = (await approval_rows(sid))[0]
        assert record.status == "cancelled" and record.resolved_at is not None
        assert fake.answers == []


async def test_health_failure_is_persisted_without_dispatch(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        from agent.engine import Health

        fake = FakeEngine()

        async def health():
            return Health(
                status="unavailable", diagnostic="engine недоступен: проверка не пройдена"
            )

        fake.health = health
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        await worker.run_job(sid, 1)
        session = (await rows(sid))[0]
        assert session.engine_state["health"]["status"] == "unavailable"
        assert "проверка" in session.engine_state["health"]["diagnostic"]
        assert not fake.sent and fake.closed.is_set()


@pytest.mark.parametrize("cause", ["ttl", "lease", "stop"])
async def test_approved_decision_is_withdrawn_before_delivery(tmp_path, monkeypatch, cause):
    async with asyncio.timeout(4):
        from agent.engine_runner import EngineRunner

        clock = Clock()
        ready, release = asyncio.Event(), asyncio.Event()
        original_init, original_answer = EngineRunner.__init__, EngineRunner.answer_request

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs, clock=clock)

        async def answer(self, request_id, decision, **kwargs):
            if decision == "allow":
                ready.set()
                await release.wait()
            return await original_answer(self, request_id, decision, **kwargs)

        monkeypatch.setattr(EngineRunner, "__init__", init)
        monkeypatch.setattr(EngineRunner, "answer_request", answer)
        fake = FakeEngine([request(), DoneEvent(status="completed", reason=None)])
        worker, sid = await setup(tmp_path, monkeypatch, fake, timeouts={"approval": 120})
        async with broker.subscribe() as queue:
            await worker.start(sid, new_run=False, expected_generation=1)
            task = worker._tasks[sid]
            try:
                event = await required(queue, task)
                await clock.waiting.wait()
                await worker.resolve_approval(
                    event["approval_id"],
                    "approve",
                    None,
                    actor_user_id="operator",
                    generation=1,
                    executor="engine",
                )
                clock.advance(1)
                await ready.wait()
                if cause == "ttl":
                    clock.advance(120)
                elif cause == "lease":
                    async with session_factory() as db:
                        await db.execute(
                            update(Session)
                            .where(Session.id == sid)
                            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
                        )
                        await db.commit()
                else:
                    await worker.stop(sid)
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                assert fake.answers == [("one", "deny")]
                record = (await approval_rows(sid))[0]
                assert record.status == "cancelled" and record.resolved_at is not None
            finally:
                release.set()
                await worker.shutdown()


async def test_native_request_closes_text_stream_and_preserves_tool_message_order(
    tmp_path, monkeypatch
):
    async with asyncio.timeout(4):
        from agent.engine import TextEvent, ToolResultEvent

        fake = FakeEngine(
            [
                TextEvent(item_id="before", text="before", phase="delta"),
                request(),
                ToolResultEvent(related_item_id="item-one", output="ok", success=True),
                TextEvent(item_id="after", text="after", phase="final"),
                DoneEvent(status="completed", reason=None),
            ]
        )
        worker, sid = await setup(tmp_path, monkeypatch, fake)
        async with broker.subscribe() as queue, broker.subscribe() as audit:
            await worker.start(sid, new_run=False, expected_generation=1)
            task = worker._tasks[sid]
            try:
                event = await required(queue, task)
                await worker.resolve_approval(
                    event["approval_id"],
                    "approve",
                    None,
                    actor_user_id="operator",
                    generation=1,
                    executor="engine",
                )
                await task
            finally:
                await worker.shutdown()
            events = []
            while not audit.empty():
                events.append(audit.get_nowait())
        _, messages, _, _ = await rows(sid)
        assert [m.content for m in messages if m.role == "assistant" and m.content] == [
            "before",
            "after",
        ]
        before = next(m for m in messages if m.role == "assistant" and m.content == "before")
        call = next(m for m in messages if m.tool_calls)
        result = next(m for m in messages if m.role == "tool")
        after = next(m for m in messages if m.role == "assistant" and m.content == "after")
        assert before.id < call.id < result.id < after.id
        stream_id = next(e["payload"]["stream_id"] for e in events if e["type"] == "message.delta")
        closed = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "message.created" and e["payload"].get("stream_id") == stream_id
        )
        required_index = next(i for i, e in enumerate(events) if e["type"] == "approval.required")
        assert closed < required_index
