"""External execution specifications, using only the neutral port and durable state."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from unittest.mock import Mock

import pytest
from sqlalchemy import select, update

from agent.config import Settings
from agent.database import session_factory
from agent.engine import (
    ApprovalEvent,
    DiffEvent,
    DoneEvent,
    ErrorEvent,
    Health,
    ReasoningEvent,
    TextEvent,
    ThreadRef,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from agent.events import broker
from agent.message_tree import append_message
from agent.models import Event, Message, Project, Session, SessionRunOutcome, ToolCall, User
from agent.runtime import AgentSupervisor
from agent.session_service import create_session_record
from tests.unit.test_engine import profile


class FakeEngine:
    def __init__(self, events=()):
        self.events = events
        self.sent = []
        self.answers = []
        self.interrupts = []
        self.closed = asyncio.Event()
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()
        self.block = False
        self.opened = []

    async def health(self):
        return Health(status="ready", supported=True, isolation_capable=True)

    async def open(self, spec, thread_id=None):
        self.opened.append((spec, thread_id))
        return ThreadRef(thread_id or "native-thread")

    async def turn(self, thread, input, submission_id):
        # The commit must be visible through an independent connection BEFORE send.
        from agent.models import EngineTurn

        async with session_factory() as db:
            turn = await db.scalar(
                select(EngineTurn).where(EngineTurn.submission_id == submission_id)
            )
            assert turn.state == "dispatched"
        self.sent.append((input, submission_id))
        self.entered.set()
        if self.block:
            await self.proceed.wait()
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield replace(
                event,
                submission_id=submission_id,
                thread_id=thread.native_id,
                turn_id="native-turn",
            )

    async def answer(self, request_id, decision):
        self.answers.append((request_id, decision))

    async def interrupt(self, turn_id):
        self.interrupts.append(turn_id)

    async def read_limits(self):
        raise AssertionError("execution must not request limits")

    async def close(self):
        self.closed.set()


async def setup(tmp_path, monkeypatch, fake, **profile_changes):
    (tmp_path / "dev.md").write_text("Instructions without Python tools")
    config = Settings(
        models_file="",
        prompts_dir=tmp_path,
        engine_profiles={"personal": profile(enabled=True, **profile_changes)},
    )
    async with session_factory() as db:
        db.add(User(id="operator", display_name="operator"))
        project = Project(name="test", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = await create_session_record(
            db,
            config,
            project,
            "work",
            "dev",
            engine_profile="personal",
            actor_user_id="operator",
            run_generation=1,
        )
        sid = session.id
    supervisor = AgentSupervisor(config)
    supervisor.engine_registry.register("personal", lambda _: fake)
    forbidden = Mock(side_effect=AssertionError("engine entered model/tool execution"))
    for name in ("_models", "_llm_for", "_run_session", "_resume_tool_calls"):
        monkeypatch.setattr(supervisor, name, forbidden)
    monkeypatch.setattr("agent.runtime.source_registry", forbidden)
    return supervisor, sid


async def rows(sid):
    from agent.models import EngineTurn

    async with session_factory() as db:
        return (
            await db.get(Session, sid),
            list(
                await db.scalars(
                    select(Message).where(Message.session_id == sid).order_by(Message.id)
                )
            ),
            list(await db.scalars(select(EngineTurn).where(EngineTurn.session_id == sid))),
            list(
                await db.scalars(
                    select(SessionRunOutcome).where(SessionRunOutcome.session_id == sid)
                )
            ),
        )


async def test_text_reasoning_final_dedupe_and_stream_identity(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [
                ReasoningEvent(item_id="reason", seq=1, text="think", phase="delta"),
                ReasoningEvent(item_id="reason", seq=2, text="thinking", phase="final"),
                TextEvent(item_id="answer", seq=3, text="hel", phase="delta"),
                TextEvent(item_id="answer", seq=4, text="hello", phase="final"),
                TextEvent(item_id="answer", seq=4, text="hello", phase="final"),
                DoneEvent(status="completed", reason=None),
            ]
        )
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        async with broker.subscribe() as queue:
            await supervisor.run_job(sid, 1)
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
        session, messages, turns, outcomes = await rows(sid)
        replies = [m for m in messages if m.role == "assistant"]
        assert [(m.content, m.reasoning_content) for m in replies] == [("hello", "thinking")]
        assert session.status == "completed" and len(outcomes) == 1
        assert len(turns) == 1 and turns[0].state == "done"
        assert turns[0].native_turn_id == "native-turn"
        assert session.engine_state["thread_id"] == "native-thread"
        assert session.engine_state["history_cursor"]
        deltas = [e for e in events if e["type"].endswith(".delta")]
        assert {e["type"] for e in deltas} == {"message.delta", "reasoning.delta"}
        assert all(
            e["payload"]["generation"] == 1
            and e["payload"]["stream_no"] > 0
            and e["payload"]["stream_id"]
            for e in deltas
        )
        assert fake.closed.is_set()
        assert fake.opened[0][0].settings["session_id"] == sid
        await supervisor.run_job(sid, 1)  # duplicate queue delivery
        assert len(fake.sent) == 1 and len((await rows(sid))[2]) == 1


async def test_native_tools_diff_usage_are_observed_only(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [
                ToolCallEvent(
                    item_id="command",
                    seq=1,
                    name="shell",
                    input={"command": "fake"},
                    status="running",
                ),
                ToolResultEvent(
                    item_id="result", seq=2, related_item_id="command", output="ok", success=True
                ),
                DiffEvent(
                    item_id="patch", seq=3, paths=["a.txt"], patch="+text", status="proposed"
                ),
                UsageEvent(
                    seq=4,
                    input_tokens=10,
                    output_tokens=4,
                    cache_tokens=2,
                    reported_cost=Decimal("0.25"),
                    estimated_cost=True,
                ),
                DoneEvent(status="completed", reason=None),
            ]
        )
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        await supervisor.run_job(sid, 1)
        session, messages, _, _ = await rows(sid)
        async with session_factory() as db:
            calls = list(await db.scalars(select(ToolCall).where(ToolCall.session_id == sid)))
            kinds = list(await db.scalars(select(Event.type).where(Event.session_id == sid)))
        assert len(calls) == 2
        assert all(c.executor == "engine" and c.native_item_id for c in calls)
        assert all(c.status == "completed" for c in calls)
        assert len([m for m in messages if m.role == "tool"]) == 2
        assert any(c.result.get("patch") == "+text" for c in calls)
        assert session.engine_state["usage"]["input_tokens"] == 10
        assert session.engine_state["usage"]["reported_cost"] == "0.25"
        assert {"engine.diff", "engine.usage", "tool.completed", "message.created"} <= set(kinds)
        assert not (tmp_path / "a.txt").exists()


@pytest.mark.parametrize(
    "events, status, reason",
    [
        ([], "error", "EOF"),
        ([RuntimeError("process crashed")], "error", "crashed"),
        ([DoneEvent(status="error", reason="failed")], "error", "failed"),
        (
            [ErrorEvent(code="protocol", message="malformed payload", retryable=False)],
            "error",
            "malformed",
        ),
        ([TextEvent(text="x" * (1024 * 1024 + 1), phase="delta")], "error", "1 MiB"),
        ([TextEvent(text="x", phase="invalid")], "error", "event"),
    ],
)
async def test_protocol_failure_is_terminal_without_retry(
    tmp_path, monkeypatch, events, status, reason
):
    async with asyncio.timeout(4):
        fake = FakeEngine(events)
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        await supervisor.run_job(sid, 1)
        session, _, turns, outcomes = await rows(sid)
        assert session.status == status
        assert reason in (outcomes[0].reason or "") + (outcomes[0].error or "")
        assert len(fake.sent) == 1 and turns[0].state == "done"
        assert fake.closed.is_set()


async def test_approval_denied_interrupts_without_waiting(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [
                ApprovalEvent(
                    item_id="request-item",
                    request_id="permission",
                    action="write",
                    parameters={},
                    scope="turn",
                ),
                DoneEvent(status="completed", reason=None),
            ]
        )
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        await supervisor.run_job(sid, 1)
        session, _, turns, outcomes = await rows(sid)
        assert fake.answers == [("permission", "deny")]
        assert fake.interrupts == ["native-turn"]
        assert session.status == "interrupted" and "разрешен" in outcomes[0].reason
        assert turns[0].outcome["status"] == "interrupted"


@pytest.mark.parametrize("stop_kind", ["stop", "lease", "shutdown"])
async def test_stop_lease_shutdown_close_and_late_done(tmp_path, monkeypatch, stop_kind):
    async with asyncio.timeout(4):
        fake = FakeEngine([DoneEvent(status="completed", reason=None)])
        fake.block = True
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        await supervisor.start(sid, new_run=False, expected_generation=1)
        task = supervisor._tasks[sid]
        await fake.entered.wait()
        if stop_kind == "stop":
            await supervisor.stop(sid)
        elif stop_kind == "shutdown":
            await supervisor.shutdown()
        else:
            async with session_factory() as db:
                await db.execute(
                    update(Session).where(Session.id == sid).values(owner_token="other")
                )
                await db.commit()
            task.cancel()  # lease keeper's cancellation, no wall clock wait
        await asyncio.gather(task, return_exceptions=True)
        assert fake.closed.is_set() and fake.interrupts
        fake.proceed.set()
        session, _, turns, outcomes = await rows(sid)
        if stop_kind == "stop":
            assert session.status == "stopped" and outcomes[0].status == "stopped"
            assert turns[0].outcome["status"] == "stopped"
        elif stop_kind == "shutdown":
            assert session.status == "interrupted"
        else:
            assert session.status != "completed"


@pytest.mark.parametrize("state", ["prepared", "dispatched"])
@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
async def test_restart_reconciles_journal_without_resending_unknown(
    tmp_path, monkeypatch, state, status
):
    async with asyncio.timeout(4):
        from agent.models import EngineTurn

        fake = FakeEngine([DoneEvent(status="completed", reason=None)])
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        async with session_factory() as db:
            await db.execute(update(Session).where(Session.id == sid).values(status=status))
            input_id = await db.scalar(
                select(Message.id).where(Message.session_id == sid, Message.role == "user")
            )
            db.add(
                EngineTurn(
                    session_id=sid,
                    generation=1,
                    input_message_id=input_id,
                    submission_id="stable-submission",
                    state=state,
                )
            )
            await db.commit()
        await supervisor.reconcile_pending()
        if sid in supervisor._tasks:
            await supervisor._tasks[sid]
        session, _, turns, outcomes = await rows(sid)
        assert len(turns) == 1
        if state == "prepared":
            assert fake.sent == [("work", "stable-submission")]
            assert session.status == "completed"
        else:
            assert fake.sent == [] and session.status == "interrupted"
            assert "результат неизвестен, проверьте файлы" in outcomes[0].reason


async def test_profile_semaphore_and_one_turn_per_session(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine([DoneEvent(status="completed", reason=None)])
        fake.block = True
        supervisor, first = await setup(tmp_path, monkeypatch, fake, process_limit=1)
        async with session_factory() as db:
            project = await db.scalar(select(Project))
            second_row = await create_session_record(
                db,
                supervisor.settings,
                project,
                "second",
                "dev",
                engine_profile="personal",
                actor_user_id="operator",
                run_generation=1,
            )
            second = second_row.id
        slot_wait = asyncio.Event()

        class ObservedSemaphore(asyncio.Semaphore):
            async def acquire(self):
                if self.locked():
                    slot_wait.set()
                return await super().acquire()

        supervisor._engine_slots["personal"] = ObservedSemaphore(1)
        first_task = asyncio.create_task(supervisor.run_job(first, 1))
        await fake.entered.wait()
        second_task = asyncio.create_task(supervisor.run_job(second, 1))
        duplicate = asyncio.create_task(supervisor.run_job(first, 1))
        await slot_wait.wait()  # second session reached the occupied profile slot
        assert len(fake.sent) == 1
        fake.proceed.set()
        await asyncio.gather(first_task, second_task, duplicate)
        assert len(fake.sent) == 2


class WorkerCrash(BaseException):
    pass


@pytest.mark.parametrize("point", ["before_dispatch", "after_dispatch", "done_before_commit"])
async def test_worker_crash_boundaries(tmp_path, monkeypatch, point):
    async with asyncio.timeout(4):
        from agent import engine_runner

        fake = FakeEngine([DoneEvent(status="completed", reason=None)])
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        if point == "before_dispatch":

            async def crash_open(*args, **kwargs):
                raise WorkerCrash()

            monkeypatch.setattr(fake, "open", crash_open)
        elif point == "after_dispatch":

            async def crash_turn(*args, **kwargs):
                raise WorkerCrash()
                yield  # async iterator

            monkeypatch.setattr(fake, "turn", crash_turn)
        else:
            real_finish = engine_runner.finish_run

            async def crash_finish(*args, **kwargs):
                await real_finish(*args, **kwargs)
                raise WorkerCrash()

            monkeypatch.setattr(engine_runner, "finish_run", crash_finish)
        with pytest.raises(WorkerCrash):
            await supervisor.run_job(sid, 1)
        session, _, turns, outcomes = await rows(sid)
        assert outcomes == [] and session.status not in {"completed", "error"}
        assert turns[0].state == ("prepared" if point == "before_dispatch" else "dispatched")
        monkeypatch.undo()
        second = FakeEngine([DoneEvent(status="completed", reason=None)])
        supervisor.engine_registry.register("personal", lambda _: second)
        await supervisor.reconcile_pending()
        if sid in supervisor._tasks:
            await supervisor._tasks[sid]
        assert bool(second.sent) == (point == "before_dispatch")


async def test_lost_fence_rejects_writes_and_deltas(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [
                TextEvent(item_id="answer", text="stale", phase="final"),
                DoneEvent(status="completed", reason=None),
            ]
        )
        fake.block = True
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        task = asyncio.create_task(supervisor.run_job(sid, 1))
        await fake.entered.wait()
        async with session_factory() as db:
            await db.execute(
                update(Session).where(Session.id == sid).values(owner_token="replacement")
            )
            await db.commit()
        async with broker.subscribe() as queue:
            fake.proceed.set()
            await task
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
        _, messages, turns, outcomes = await rows(sid)
        assert not [m for m in messages if m.role == "assistant"]
        assert not [e for e in events if e["type"].endswith(".delta")]
        assert outcomes == [] and turns[0].state == "dispatched"
        assert fake.interrupts and fake.closed.is_set()


async def test_interleaved_items_and_partial_recovery(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [
                TextEvent(item_id="first", seq=1, text="one", phase="delta"),
                TextEvent(item_id="second", seq=2, text="two", phase="delta"),
                TextEvent(item_id="first", seq=3, text="ONE", phase="final"),
                RuntimeError("crashed"),
            ]
        )
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        await supervisor.run_job(sid, 1)
        session, messages, turns, outcomes = await rows(sid)
        answer = next(m for m in messages if m.role == "assistant")
        assert answer.content == "ONEtwo" and answer.kind == "interrupted"
        assert outcomes[0].status == session.status == "error"
        assert turns[0].outcome["status"] == "error"


async def test_first_finish_wins_even_when_engine_ignores_cancellation(tmp_path, monkeypatch):
    async with asyncio.timeout(4):

        class LateEngine(FakeEngine):
            async def turn(self, thread, input, submission_id):
                self.entered.set()
                try:
                    await self.proceed.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await late.wait()
                yield DoneEvent(turn_id="late-turn", status="completed", reason=None)

        cancelled, late = asyncio.Event(), asyncio.Event()
        fake = LateEngine()
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        await supervisor.start(sid, new_run=False, expected_generation=1)
        await fake.entered.wait()
        stopper = asyncio.create_task(supervisor.stop(sid))
        await cancelled.wait()  # Stop's transaction committed before task cancellation
        late.set()
        await stopper
        await supervisor._tasks[sid]
        session, _, turns, outcomes = await rows(sid)
        assert session.status == "stopped" and [o.status for o in outcomes] == ["stopped"]
        assert turns[0].state == "done" and turns[0].outcome["status"] == "stopped"


@pytest.mark.parametrize(
    "status", ["pending", "running", "awaiting_confirmation", "awaiting_child"]
)
@pytest.mark.parametrize("live", [False, True])
async def test_recovery_cas_does_not_steal_live_owner(tmp_path, monkeypatch, status, live):
    async with asyncio.timeout(4):
        from datetime import UTC, datetime, timedelta

        from agent.database import init_database
        from agent.models import EngineTurn

        fake = FakeEngine()
        _, sid = await setup(tmp_path, monkeypatch, fake)
        async with session_factory() as db:
            session = await db.get(Session, sid)
            session.status = status
            session.owner_token = "worker"
            session.lease_until = datetime.now(UTC) + timedelta(seconds=60 if live else -60)
            input_id = await db.scalar(
                select(Message.id).where(Message.session_id == sid, Message.role == "user")
            )
            db.add(
                EngineTurn(
                    session_id=sid, generation=1, input_message_id=input_id, state="dispatched"
                )
            )
            await db.commit()
        await init_database(run_migrations=False, recover_running=True)
        await init_database(run_migrations=False, recover_running=True)
        session, _, turns, outcomes = await rows(sid)
        if live:
            assert outcomes == [] and session.status == status and session.owner_token == "worker"
        else:
            assert session.status == "interrupted" and session.run_generation == 2
            assert len(outcomes) == 1 and turns[0].outcome["status"] == "interrupted"


async def test_cancelled_database_frame_releases_writer_before_cleanup(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        from agent import engine_runner

        entered = asyncio.Event()
        original = engine_runner.add_event

        async def block(db, kind, payload, session_id):
            event = await original(db, kind, payload, session_id)
            if kind == "message.created":
                entered.set()
                await asyncio.Event().wait()
            return event

        fake = FakeEngine([TextEvent(item_id="answer", text="partial", phase="final")])
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        monkeypatch.setattr(engine_runner, "add_event", block)
        task = asyncio.create_task(supervisor.run_job(sid, 1))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session, messages, turns, outcomes = await rows(sid)
        assert session.status == "interrupted" and len(outcomes) == 1
        assert not [m for m in messages if m.role == "assistant"]  # item transaction rolled back
        assert turns[0].outcome["status"] == "interrupted"


@pytest.mark.parametrize("ending", ["done", "eof", "crash", "stop", "malformed", "oversize"])
async def test_runner_with_guarded_fake_cli(tmp_path, monkeypatch, ending):
    async with asyncio.timeout(5):
        import json
        import sys

        from agent.engine_process import GuardedProcess, decode_line

        script = tmp_path / "cli.py"
        script.write_text("""import json, os, signal, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
request = json.loads(sys.stdin.readline())
# Only stdio is inherited; no private watchdog or status descriptors.
fds = []
for fd in range(3, 256):
    try:
        os.fstat(fd)
        fds.append(fd)
    except OSError:
        pass
print(json.dumps({'ready': True, 'fds': fds}), flush=True)
ending = sys.argv[1]
if ending == 'stop':
    signal.pause()
print(json.dumps({'text': request['input']}), flush=True)
if ending == 'done':
    print(json.dumps({'done': True}), flush=True)
if ending == 'crash':
    sys.exit(7)
if ending == 'malformed':
    print('{', flush=True)
if ending == 'oversize':
    print('x' * (1024 * 1024 + 1), flush=True)
""")

        class ProcessEngine(FakeEngine):
            async def open(self, spec, thread_id=None):
                self.process = await GuardedProcess.start(
                    [sys.executable, str(script), ending],
                    cwd=tmp_path,
                    lock_path=tmp_path / (spec.settings["session_id"] + ".lock"),
                )
                return ThreadRef("process-thread")

            async def turn(self, thread, input, submission_id):
                from agent.models import EngineTurn

                async with session_factory() as db:
                    row = await db.scalar(
                        select(EngineTurn).where(EngineTurn.submission_id == submission_id)
                    )
                    assert row.state == "dispatched"
                self.process.stdin.write(json.dumps({"input": input}).encode() + b"\n")
                await self.process.stdin.drain()
                ready = decode_line(await self.process.stdout.readline())
                assert ready == {"ready": True, "fds": []}
                self.entered.set()
                while raw := await self.process.stdout.readline():
                    payload = decode_line(raw)
                    if payload.get("done"):
                        yield DoneEvent(turn_id="process-turn", status="completed", reason=None)
                    else:
                        yield TextEvent(
                            turn_id="process-turn",
                            item_id="reply",
                            text=payload["text"],
                            phase="final",
                        )

            async def close(self):
                if hasattr(self, "process"):
                    await self.process.close()
                await super().close()

        fake = ProcessEngine()
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        try:
            await supervisor.start(sid, new_run=False, expected_generation=1)
            await fake.entered.wait()
            if ending == "stop":
                await supervisor.stop(sid)
            await asyncio.gather(supervisor._tasks[sid], return_exceptions=True)
            session, messages, turns, outcomes = await rows(sid)
            expected = (
                "completed" if ending == "done" else "stopped" if ending == "stop" else "error"
            )
            assert session.status == expected and outcomes[0].status == expected
            assert turns[0].state == "done" and fake.process.returncode is not None
        finally:
            await supervisor.shutdown()
            await fake.close()


async def test_new_turn_waits_for_old_engine_close(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        from agent.session_fence import new_generation

        closing, release_close, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class SlowClose(FakeEngine):
            async def close(self):
                closing.set()
                await release_close.wait()
                await super().close()

        class ObservedLock(asyncio.Lock):
            async def acquire(self):
                if self.locked():
                    waiting.set()
                return await super().acquire()

        first = SlowClose()
        first.block = True
        second = FakeEngine([DoneEvent(status="completed", reason=None)])
        supervisor, sid = await setup(tmp_path, monkeypatch, first, process_limit=2)
        engines = iter((first, second))
        supervisor.engine_registry.register("personal", lambda _: next(engines))
        supervisor._engine_sessions[sid] = ObservedLock()
        await supervisor.start(sid, new_run=False, expected_generation=1)
        await first.entered.wait()
        old_task = supervisor._tasks[sid]
        stopper = asyncio.create_task(supervisor.stop(sid))
        await closing.wait()
        async with session_factory() as db:
            await new_generation(db, sid)
            await db.execute(update(Session).where(Session.id == sid).values(status="pending"))
            await append_message(db, sid, role="user", content="next")
            await db.commit()
        await supervisor.start(sid, new_run=False, expected_generation=3)
        await waiting.wait()
        assert not second.opened and not first.closed.is_set()
        release_close.set()
        await asyncio.gather(stopper, old_task, supervisor._tasks[sid], return_exceptions=True)
        assert first.closed.is_set() and len(second.sent) == 1
        session, _, _, outcomes = await rows(sid)
        assert session.status == "completed" and len(outcomes) == 2


@pytest.mark.parametrize("role", ["ask", "arch", "cybersec", "analyst"])
async def test_read_roles_keep_native_read_mode_and_no_python_tool_promise(
    tmp_path, monkeypatch, role
):
    async with asyncio.timeout(4):
        fake = FakeEngine([DoneEvent(status="completed", reason=None)])
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        async with session_factory() as db:
            await db.execute(update(Session).where(Session.id == sid).values(mode=role))
            await db.commit()
        await supervisor.run_job(sid, 1)
        spec, _ = fake.opened[0]
        assert spec.mode == "read"
        assert "собственные инструменты" in spec.instructions
        assert "Instructions without Python tools" in spec.instructions


async def test_durable_stream_closure_matches_existing_message_transport(tmp_path, monkeypatch):
    async with asyncio.timeout(4):
        fake = FakeEngine(
            [
                TextEvent(item_id="first", text="before", phase="final"),
                ToolCallEvent(item_id="tool", name="shell", input={}, status="running"),
                ToolResultEvent(related_item_id="tool", output="ok", success=True),
                TextEvent(item_id="second", text="after", phase="delta"),
                TextEvent(item_id="second", text="after!", phase="final"),
                DoneEvent(status="completed", reason=None),
            ]
        )
        supervisor, sid = await setup(tmp_path, monkeypatch, fake)
        async with broker.subscribe() as queue:
            await supervisor.run_job(sid, 1)
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
        closed = set()
        for event in events:
            payload = event["payload"]
            if event["type"].endswith(".delta"):
                assert payload["stream_id"] not in closed
            if event["type"] == "message.created" and "stream_id" in payload:
                assert payload["role"] == "assistant"
                closed.add(payload["stream_id"])
        assert len(closed) == 2
