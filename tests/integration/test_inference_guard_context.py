"""Logical inference scope and terminal guards through context/summary paths."""

import asyncio
from dataclasses import replace

import pytest

from agent.llm import LLMError
from tests.inference_guard_helpers import guard_tasks as guard_tasks
from tests.integration.test_context_inference import (
    _add_messages,
    _create_session,
    _inference,
    _model,
    _registry,
    _simple_history,
)
from tests.unit.test_inference_guard_deadline import Clock

pytestmark = pytest.mark.usefixtures("guard_tasks")


async def test_prepare_is_inside_the_only_deadline(tmp_path, monkeypatch):
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        model = _model("primary")
        session_id = await _create_session(tmp_path, "prepare-deadline")
        inference = _inference(
            session_id, _registry(model), lambda _: pytest.fail("send"), primary=model
        )
        entered = asyncio.Event()

        async def prepare(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(inference, "prepare", prepare)
        clock = Clock()
        guard = GuardContext(model, clock=clock)
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(guard.run(inference.run(model)))
            await entered.wait()
            await clock.armed.get()
            clock.advance(600)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_inference_timeout"
            assert inference.state.http_attempts == 0
        finally:
            CURRENT_GUARD.reset(token)


@pytest.mark.parametrize(
    "kind",
    [
        "llm_inference_timeout",
        "llm_semantic_idle",
        "llm_empty_response",
        "llm_repetition",
        "llm_invalid_response",
    ],
)
async def test_guards_never_fallback(tmp_path, monkeypatch, kind):
    async with asyncio.timeout(5):
        from agent.llm_errors import LLMErrorKind, ProviderErrorInfo

        model, fallback = _model("primary"), _model("secondary")
        session_id = await _create_session(tmp_path, "terminal")
        await _add_messages(session_id, _simple_history())
        sent = []
        inference = _inference(
            session_id, _registry(model, (fallback,)), lambda _: None, primary=model
        )

        async def send(candidate, *args, **kwargs):
            sent.append(candidate.id)
            raise LLMError(
                "guard", info=ProviderErrorInfo(LLMErrorKind(kind), None, kind, "guard", False)
            )

        monkeypatch.setattr(inference, "_send", send)
        with pytest.raises(LLMError) as caught:
            await inference.run(model)
        assert caught.value.kind.value == kind
        assert sent == ["primary"]


@pytest.mark.parametrize(
    "kind",
    [
        "llm_inference_timeout",
        "llm_semantic_idle",
        "llm_empty_response",
        "llm_repetition",
        "llm_invalid_response",
        "cancelled",
    ],
)
async def test_summary_reraises_every_terminal_guard(tmp_path, monkeypatch, kind):
    async with asyncio.timeout(5):
        from agent.llm_errors import LLMErrorKind, ProviderErrorInfo

        model = _model("primary")
        session_id = await _create_session(tmp_path, "summary-guard")
        await _add_messages(session_id, _simple_history())
        inference = _inference(session_id, _registry(model), lambda _: None, primary=model)

        async def send(*args, **kwargs):
            if kind == "cancelled":
                raise asyncio.CancelledError
            raise LLMError(
                "guard", info=ProviderErrorInfo(LLMErrorKind(kind), None, kind, "guard", False)
            )

        async def prepare(*args, summary_runner, **kwargs):
            await summary_runner([{"role": "user", "content": "summarize"}])
            pytest.fail("summary guard was hidden; main inference could continue")

        monkeypatch.setattr(inference, "_send", send)
        monkeypatch.setattr(inference.manager, "prepare", prepare)
        if kind == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await inference.prepare(model, inference.reasoning_for(model))
        else:
            with pytest.raises(LLMError) as caught:
                await inference.prepare(model, inference.reasoning_for(model))
            assert caught.value.kind.value == kind


async def test_substituted_client_rejected_before_normal_message_or_any_tool(tmp_path):
    async with asyncio.timeout(5):
        from sqlalchemy import select

        from agent.config import Settings
        from agent.database import session_factory
        from agent.llm import LLMResponse
        from agent.models import Message, ToolCall
        from agent.runtime import AgentSupervisor

        session_id = await _create_session(tmp_path, "invalid-batch")
        await _add_messages(session_id, _simple_history())

        class Client:
            calls = 0

            async def chat(self, *args, **kwargs):
                self.calls += 1
                if self.calls > 1:
                    pytest.fail("invalid batch was accepted and retried")
                return LLMResponse(
                    None,
                    [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                        },
                        {
                            "type": "function",
                            "function": {"name": "write_file", "arguments": '{"path":5}'},
                        },
                    ],
                    "tool_calls",
                    1,
                    1,
                )

        supervisor = AgentSupervisor(Settings())
        supervisor.llm = Client()
        await supervisor._run_session(session_id)
        async with session_factory() as db:
            from agent.models import Session, SessionRunOutcome

            row = await db.get(Session, session_id)
            assert row.status == "error"
            outcome = await db.get(SessionRunOutcome, (session_id, row.run_generation))
            assert outcome.reason == "llm_invalid_response"
            assert (
                list(await db.scalars(select(ToolCall).where(ToolCall.session_id == session_id)))
                == []
            )
            assert (
                list(
                    await db.scalars(
                        select(Message).where(
                            Message.session_id == session_id,
                            Message.role == "assistant",
                            Message.kind == "normal",
                        )
                    )
                )
                == []
            )


async def test_late_start_audit_finishes_same_attempt_with_first_reason(tmp_path, monkeypatch):
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext
        from tests.integration.test_context_inference import _audit_rows

        session_id = await _create_session(tmp_path, "late-audit")
        await _add_messages(session_id, _simple_history())
        model = _model("primary")
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        from agent import context_inference as module
        from tests.integration.test_context_inference import _make_client_factory

        real_start, real_finish = module.start_attempt, module.finish_attempt
        terminals = []

        async def start(*args, **kwargs):
            started.set()
            await release.wait()
            return await real_start(*args, **kwargs)

        async def finish(audit_id, record, key):
            terminals.append((audit_id, record))
            await real_finish(audit_id, record, key)
            finished.set()

        monkeypatch.setattr(module, "start_attempt", start)
        monkeypatch.setattr(module, "finish_attempt", finish)
        inference = _inference(
            session_id,
            _registry(model),
            _make_client_factory(lambda _: pytest.fail("late HTTP")),
            primary=model,
        )
        clock = Clock()
        guard = GuardContext(model, clock=clock)
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(guard.run(inference.run(model)))
            await started.wait()
            clock.advance(600)
            with pytest.raises(LLMError):
                await task
            release.set()
            await finished.wait()
            audits = await _audit_rows(session_id)
            assert len(audits) == 1 and len(terminals) == 1
            assert audits[0].error_kind == "llm_inference_timeout"
            assert audits[0].prompt_tokens is None
            assert audits[0].completion_tokens is None
        finally:
            CURRENT_GUARD.reset(token)


@pytest.mark.parametrize("point", ["prepare", "checkpoint", "between_retries", "result_check"])
async def test_runtime_hangs_share_one_deadline(tmp_path, monkeypatch, point):
    async with asyncio.timeout(5):
        from sqlalchemy.ext.asyncio import AsyncSession

        from agent import context_inference as context_module
        from agent import runtime as runtime_module
        from agent.config import Settings
        from agent.database import session_factory
        from agent.inference_guard import GuardContext
        from agent.llm import LLMResponse
        from agent.llm_errors import LLMErrorKind, ProviderErrorInfo
        from agent.models import Session, SessionRunOutcome
        from agent.runtime import AgentSupervisor

        clock = Clock()
        entered = asyncio.Event()
        model = _model("primary")
        session_id = await _create_session(tmp_path, "runtime-hang")
        await _add_messages(session_id, _simple_history())
        supervisor = AgentSupervisor(Settings())
        sent = []

        class Client:
            async def chat(self, *args, **kwargs):
                sent.append(1)
                if point == "between_retries":
                    raise LLMError(
                        "incomplete",
                        info=ProviderErrorInfo(
                            LLMErrorKind.stream_incomplete, None, None, "incomplete", True
                        ),
                        started=True,
                    )
                return LLMResponse("answer", [], "stop", 1, 1)

        supervisor.llm = Client()

        async def models(db):
            return _registry(model)

        monkeypatch.setattr(supervisor, "_models", models)
        monkeypatch.setattr(
            runtime_module,
            "GuardContext",
            lambda candidate, **kwargs: GuardContext(candidate, clock=clock, **kwargs),
        )
        real_prepare = context_module.ContextInference.prepare

        async def prepare(self, *args, **kwargs):
            if point == "prepare":
                entered.set()
                await asyncio.Event().wait()
            result = await real_prepare(self, *args, **kwargs)
            if point == "checkpoint":
                return replace(result, context=replace(result.context, checkpoint_to_save=object()))
            return result

        monkeypatch.setattr(context_module.ContextInference, "prepare", prepare)
        if point == "checkpoint":

            async def checkpoint(*args, **kwargs):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(context_module, "persist_checkpoint", checkpoint)
        if point == "between_retries":
            original_event = runtime_module.emit_event

            async def event(db, kind, *args, **kwargs):
                if kind == "inference.retry":
                    entered.set()
                    await asyncio.Event().wait()
                return await original_event(db, kind, *args, **kwargs)

            monkeypatch.setattr(runtime_module, "emit_event", event)
        if point == "result_check":
            original_refresh = AsyncSession.refresh

            async def refresh(self, instance, attribute_names=None, **kwargs):
                if attribute_names == ["status", "stop_requested"]:
                    entered.set()
                    await asyncio.Event().wait()
                return await original_refresh(
                    self, instance, attribute_names=attribute_names, **kwargs
                )

            monkeypatch.setattr(AsyncSession, "refresh", refresh)
        task = asyncio.create_task(supervisor._run_session(session_id))
        barrier = asyncio.create_task(entered.wait())
        done, _ = await asyncio.wait({task, barrier}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
            pytest.fail("Inference ended before reaching the required barrier")
        clock.advance(600)
        await task
        assert len(sent) <= 1
        async with session_factory() as db:
            row = await db.get(Session, session_id)
            outcome = await db.get(SessionRunOutcome, (session_id, row.run_generation))
            assert row.status == "error"
            assert outcome.reason == "llm_inference_timeout"


async def test_stalled_terminal_audit_keeps_first_reason_and_reported_usage(tmp_path, monkeypatch):
    async with asyncio.timeout(5):
        from agent import context_inference as module
        from agent.inference_guard import CURRENT_GUARD, GuardContext
        from tests.integration.test_context_inference import (
            _audit_rows,
            _make_client_factory,
            _stream_response,
        )

        session_id = await _create_session(tmp_path, "finish-audit")
        await _add_messages(session_id, _simple_history())
        model = _model("primary")
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        real_finish = module.finish_attempt

        async def finish(audit_id, record, key):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            await real_finish(audit_id, record, key)
            finished.set()

        monkeypatch.setattr(module, "finish_attempt", finish)
        import httpx

        inference = _inference(
            session_id,
            _registry(model),
            _make_client_factory(
                lambda _: httpx.Response(
                    200, content=_stream_response({"prompt_tokens": 7, "completion_tokens": 3})
                )
            ),
            primary=model,
        )
        clock = Clock()
        guard = GuardContext(model, clock=clock)
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(guard.run(inference.run(model)))
            await entered.wait()
            clock.advance(600)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_inference_timeout"
            release.set()
            await finished.wait()
            audits = await _audit_rows(session_id)
            assert len(audits) == 1
            assert audits[0].status == "error"
            assert audits[0].error_kind == "llm_inference_timeout"
            assert (audits[0].prompt_tokens, audits[0].completion_tokens) == (7, 3)
        finally:
            CURRENT_GUARD.reset(token)


async def test_summary_guard_survives_stuck_close(tmp_path, monkeypatch):
    async with asyncio.timeout(5):
        from agent import context_inference as module
        from agent.inference_guard import CURRENT_GUARD, GuardContext, guard_error
        from agent.llm_errors import LLMErrorKind

        model = _model("primary")
        session_id = await _create_session(tmp_path, "summary-close")
        await _add_messages(session_id, _simple_history())
        inference = _inference(session_id, _registry(model), lambda _: None, primary=model)
        entered, release = asyncio.Event(), asyncio.Event()

        class Client:
            def __init__(self, model):
                pass

            async def aclose(self):
                entered.set()
                await release.wait()

        monkeypatch.setattr(module, "LLMClient", Client)

        async def send(*args, **kwargs):
            raise guard_error(LLMErrorKind.repetition)

        async def prepare(*args, summary_runner, **kwargs):
            await summary_runner([{"role": "user", "content": "summarize"}])
            pytest.fail("main inference continued")

        monkeypatch.setattr(inference, "_send", send)
        monkeypatch.setattr(inference.manager, "prepare", prepare)
        clock = Clock()
        guard = GuardContext(model, clock=clock)
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(
                guard.run(inference.prepare(model, inference.reasoning_for(model)))
            )
            await entered.wait()
            # Close cannot hide the guard until the total timeout.
            assert guard.closed
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_repetition"
            release.set()
        finally:
            CURRENT_GUARD.reset(token)


async def test_incomplete_runtime_recovery_is_exactly_three_calls(tmp_path):
    async with asyncio.timeout(5):
        from agent.config import Settings
        from agent.database import session_factory
        from agent.llm_errors import LLMErrorKind, ProviderErrorInfo
        from agent.models import Session, SessionRunOutcome
        from agent.runtime import AgentSupervisor

        session_id = await _create_session(tmp_path, "incomplete-runtime")
        await _add_messages(session_id, _simple_history())

        class Client:
            calls = 0

            async def chat(self, *args, **kwargs):
                self.calls += 1
                raise LLMError(
                    "interrupted",
                    info=ProviderErrorInfo(
                        LLMErrorKind.stream_incomplete, None, None, "interrupted", True
                    ),
                    started=True,
                )

        client = Client()
        supervisor = AgentSupervisor(Settings())
        supervisor.llm = client
        await supervisor._run_session(session_id)
        assert client.calls == 3
        async with session_factory() as db:
            row = await db.get(Session, session_id)
            outcome = await db.get(SessionRunOutcome, (session_id, row.run_generation))
            assert outcome.reason == "llm_stream_incomplete"


@pytest.mark.parametrize("boundary", ["stop", "session_deadline"])
async def test_stop_audit_preserves_usage_from_partial_stream(tmp_path, monkeypatch, boundary):
    async with asyncio.timeout(5):
        import httpx

        from agent import context_inference as module
        from agent.inference_guard import CURRENT_GUARD, GuardContext
        from tests.integration.test_context_inference import _audit_rows, _make_client_factory
        from tests.unit.test_llm_send import _sse

        session_id = await _create_session(tmp_path, "stop-usage")
        await _add_messages(session_id, _simple_history())
        model = _model("primary")
        reading, finished = asyncio.Event(), asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _sse(
                    {
                        "choices": [{"delta": {"content": "partial"}}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    }
                )
                reading.set()
                await asyncio.Event().wait()

        real_finish = module.finish_attempt

        async def finish(*args):
            await real_finish(*args)
            finished.set()

        monkeypatch.setattr(module, "finish_attempt", finish)
        inference = _inference(
            session_id,
            _registry(model),
            _make_client_factory(lambda _: httpx.Response(200, stream=Stream())),
            primary=model,
        )
        clock = Clock()
        guard = GuardContext(
            model, clock=clock, session_deadline=30 if boundary == "session_deadline" else None
        )
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(guard.run(inference.run(model)))
            await reading.wait()
            if boundary == "stop":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                from agent.session_budget_store import BudgetExceeded

                clock.advance(30)
                with pytest.raises(BudgetExceeded):
                    await task
            await finished.wait()
            audits = await _audit_rows(session_id)
            assert len(audits) == 1
            assert audits[0].status == ("cancelled" if boundary == "stop" else "error")
            assert audits[0].error_kind == (
                "cancelled" if boundary == "stop" else "deadline_exceeded"
            )
            assert (audits[0].prompt_tokens, audits[0].completion_tokens) == (7, 3)
        finally:
            CURRENT_GUARD.reset(token)


async def test_inference_database_resource_stays_with_its_worker(tmp_path, monkeypatch):
    async with asyncio.timeout(5):
        from sqlalchemy.ext.asyncio import AsyncSession

        from agent.config import Settings
        from agent.database import session_factory
        from agent.llm import LLMResponse
        from agent.models import Session
        from agent.runtime import AgentSupervisor

        session_id = await _create_session(tmp_path, "database-owner")
        await _add_messages(session_id, _simple_history())
        original_enter, original_refresh = AsyncSession.__aenter__, AsyncSession.refresh

        async def enter(self):
            result = await original_enter(self)
            self.info["test_owner"] = asyncio.current_task()
            return result

        async def refresh(self, instance, attribute_names=None, **kwargs):
            if attribute_names == ["status", "stop_requested"]:
                assert self.info["test_owner"] is asyncio.current_task(), (
                    "inference shares a session with the observer"
                )
            return await original_refresh(self, instance, attribute_names=attribute_names, **kwargs)

        monkeypatch.setattr(AsyncSession, "__aenter__", enter)
        monkeypatch.setattr(AsyncSession, "refresh", refresh)

        class Client:
            async def chat(self, *args, **kwargs):
                return LLMResponse("answer", [], "stop", 1, 1)

        supervisor = AgentSupervisor(Settings())
        supervisor.llm = Client()
        await supervisor._run_session(session_id)
        async with session_factory() as db:
            assert (await db.get(Session, session_id)).status == "completed"


@pytest.mark.parametrize("prior_guard", [False, True])
async def test_run_release_cannot_hold_stop_past_cleanup_deadline(
    tmp_path, monkeypatch, prior_guard
):
    async with asyncio.timeout(5):
        from agent import runtime as module
        from agent.config import Settings
        from agent.runtime import AgentSupervisor

        clock = Clock()
        session_id = await _create_session(tmp_path, "bounded-release")
        supervisor = AgentSupervisor(Settings())
        entered, releasing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def run(session_id):
            if prior_guard:
                from agent.inference_guard import CURRENT_RUN_GUARD, GuardContext

                guard = GuardContext(_model("primary"), clock=clock)
                guard.close(asyncio.CancelledError())
                CURRENT_RUN_GUARD.set(guard)
            entered.set()
            await asyncio.Event().wait()

        async def release_fence(*args, **kwargs):
            releasing.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

        monkeypatch.setattr(supervisor, "_run", run)
        monkeypatch.setattr(module, "release", release_fence)
        monkeypatch.setattr(module, "LoopClock", lambda: clock, raising=False)
        task = asyncio.create_task(supervisor._execute_run(session_id, None))
        try:
            await entered.wait()
            if prior_guard:
                clock.advance(1)
            task.cancel()
            await releasing.wait()
            assert await clock.armed.get() == 2
            clock.advance(2)
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            from contextlib import suppress

            with suppress(asyncio.CancelledError):
                await task


async def test_late_start_terminal_is_bounded_and_keeps_observed_usage(tmp_path, monkeypatch):
    async with asyncio.timeout(5):
        from agent import context_inference as module
        from agent.calibration_store import Calibration
        from agent.inference_guard import GuardContext
        from tests.unit.test_llm_send import _request

        clock = Clock()
        model = _model("primary")
        guard = GuardContext(model, clock=clock)
        started, release, terminal, cancelled = (asyncio.Event() for _ in range(4))
        session_id = await _create_session(tmp_path, "bounded-late-start")
        inference = _inference(session_id, _registry(model), lambda _: None, primary=model)
        records = []

        async def start(*args, **kwargs):
            started.set()
            await release.wait()
            return 42

        async def finish(audit_id, record, key):
            records.append((audit_id, record))
            terminal.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        class Client:
            async def send(self, request, *, on_attempt, **kwargs):
                from agent.llm import AttemptRecord

                await on_attempt(AttemptRecord(0, "started", 1, "hash"))
                pytest.fail("HTTP after close")

        monkeypatch.setattr(module, "start_attempt", start)
        monkeypatch.setattr(module, "finish_attempt", finish)
        task = asyncio.create_task(
            guard.run(
                inference._send(
                    model, Client(), _request(model, stream=False), Calibration(1.0, 0), "inference"
                )
            )
        )
        try:
            await started.wait()
            guard.observe_usage(7, 3)
            clock.advance(600)
            with pytest.raises(LLMError):
                await task
            release.set()
            await terminal.wait()
            assert records[0][0] == 42
            assert (records[0][1].prompt_tokens, records[0][1].completion_tokens) == (7, 3)
            clock.advance(602)
            await cancelled.wait()
            assert len(records) == 1
        finally:
            release.set()
            task.cancel()


@pytest.mark.parametrize("action", ["stop", "scoped_stop", "shutdown"])
async def test_stop_and_shutdown_do_not_join_an_uncooperative_run(tmp_path, monkeypatch, action):
    async with asyncio.timeout(5):
        from agent import runtime as module
        from agent.config import Settings
        from agent.database import session_factory
        from agent.models import Session
        from agent.runtime import AgentSupervisor

        clock = Clock()
        monkeypatch.setattr(module, "LoopClock", lambda: clock)
        session_id = await _create_session(tmp_path, "bounded-stop")
        async with session_factory() as db:
            generation = (await db.get(Session, session_id)).run_generation
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def run():
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()

        supervisor = AgentSupervisor(Settings())
        worker = asyncio.create_task(run())
        supervisor._tasks[session_id] = worker
        supervisor._task_generation[session_id] = generation
        await entered.wait()
        if action == "shutdown":
            stopping = asyncio.create_task(supervisor.shutdown())
        elif action == "stop":
            stopping = asyncio.create_task(supervisor.stop(session_id))
        else:
            stopping = asyncio.create_task(supervisor.stop_scoped(session_id, generation))
        try:
            await cancelled.wait()
            assert not clock.armed.empty(), "Stop waits without a cleanup boundary"
            assert clock.armed.get_nowait() == 2
            clock.advance(2)
            await stopping
            assert not worker.done()
        finally:
            release.set()
            await worker
            await stopping


@pytest.mark.parametrize("recovery", ["output_increase", "reasoning", "outer_cutoff"])
async def test_all_output_recoveries_keep_the_original_deadline(tmp_path, monkeypatch, recovery):
    async with asyncio.timeout(5):
        from datetime import UTC, datetime

        from sqlalchemy import select

        from agent import runtime as module
        from agent.config import Settings
        from agent.database import session_factory
        from agent.inference_guard import CURRENT_GUARD, GuardContext
        from agent.llm import LLMResponse
        from agent.model_config import ReasoningSpec
        from agent.models import Message, Session, SessionRunOutcome, ToolCall
        from agent.runtime import AgentSupervisor
        from agent.session_budget_store import load_budget_usage

        model = replace(
            _model("primary"),
            reasoning=ReasoningSpec(
                supported=True,
                allowed_efforts=["low", "high"],
                default_effort="high",
                wire_parameter="reasoning_effort",
            ),
        )
        clock = Clock()
        entered = asyncio.Event()
        guards, limits, efforts = [], [], []
        session_id = await _create_session(tmp_path, "shared-output-deadline")
        await _add_messages(session_id, _simple_history())

        class Client:
            async def chat(self, *args, **kwargs):
                guards.append(CURRENT_GUARD.get())
                limits.append(kwargs["max_tokens"])
                efforts.append(kwargs["reasoning_effort"])
                if len(guards) == 2:
                    entered.set()
                    await asyncio.Event().wait()
                clock.advance(100)
                calls = (
                    [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":'},
                        }
                    ]
                    if recovery == "outer_cutoff"
                    else []
                )
                return LLMResponse(
                    None,
                    calls,
                    "length",
                    1,
                    1,
                    reasoning="still thinking" if recovery == "reasoning" else None,
                )

        supervisor = AgentSupervisor(Settings())
        supervisor.llm = Client()

        async def models(db):
            return _registry(model)

        monkeypatch.setattr(supervisor, "_models", models)
        monkeypatch.setattr(
            module,
            "GuardContext",
            lambda candidate, **kw: GuardContext(candidate, clock=clock, **kw),
        )
        task = asyncio.create_task(supervisor._run_session(session_id))
        try:
            await entered.wait()
            assert guards[0] is guards[1]
            assert guards[0].deadline == 600
            if recovery == "output_increase":
                assert limits[1] > limits[0]
            if recovery == "reasoning":
                assert efforts == ["high", "low"]
            clock.advance(600)
            await task
            async with session_factory() as db:
                row = await db.get(Session, session_id)
                outcome = await db.get(SessionRunOutcome, (session_id, row.run_generation))
                assert outcome.reason == "llm_inference_timeout"
                assert not list(
                    await db.scalars(select(ToolCall).where(ToolCall.session_id == session_id))
                )
                assert not list(
                    await db.scalars(
                        select(Message).where(
                            Message.session_id == session_id,
                            Message.role == "assistant",
                            Message.kind == "normal",
                        )
                    )
                )
                usage = await load_budget_usage(db, row, datetime.now(UTC))
                assert usage.steps == (2 if recovery == "outer_cutoff" else 1)
        finally:
            task.cancel()


@pytest.mark.parametrize(
    "kind",
    ["inference_timeout", "semantic_idle", "empty_response", "repetition", "invalid_response"],
)
async def test_runtime_guard_has_no_completed_audit_message_or_tool(tmp_path, monkeypatch, kind):
    async with asyncio.timeout(5):
        from sqlalchemy import select

        from agent.config import Settings
        from agent.database import session_factory
        from agent.inference_guard import guard_error
        from agent.llm_errors import LLMErrorKind
        from agent.models import Message, OutboundAudit, Session, SessionRunOutcome, ToolCall
        from agent.runtime import AgentSupervisor

        model, secondary = _model("primary"), _model("secondary")
        session_id = await _create_session(tmp_path, "terminal-runtime")
        await _add_messages(session_id, _simple_history())
        calls = []
        from agent import context_inference as context_module

        finished = asyncio.Event()
        original_finish = context_module.finish_attempt

        async def finish(*args, **kwargs):
            await original_finish(*args, **kwargs)
            finished.set()

        monkeypatch.setattr(context_module, "finish_attempt", finish)

        class Client:
            async def chat(self, *args, **kwargs):
                calls.append(1)
                raise guard_error(LLMErrorKind[kind])

        supervisor = AgentSupervisor(Settings())
        supervisor.llm = Client()

        async def models(db):
            return _registry(model, (secondary,))

        monkeypatch.setattr(supervisor, "_models", models)
        monkeypatch.setattr(supervisor, "_llm_for", lambda models, candidate: supervisor.llm)
        await supervisor._run_session(session_id)
        assert calls == [1]
        await finished.wait()
        async with session_factory() as db:
            row = await db.get(Session, session_id)
            outcome = await db.get(SessionRunOutcome, (session_id, row.run_generation))
            assert row.status == "error" and outcome.reason == LLMErrorKind[kind].value
            audits = list(
                await db.scalars(
                    select(OutboundAudit).where(OutboundAudit.session_id == session_id)
                )
            )
            assert len(audits) == 1 and audits[0].status == "error"
            assert audits[0].error_kind == outcome.reason
            assert not list(
                await db.scalars(select(ToolCall).where(ToolCall.session_id == session_id))
            )
            assert not list(
                await db.scalars(
                    select(Message).where(
                        Message.session_id == session_id,
                        Message.role == "assistant",
                        Message.kind == "normal",
                    )
                )
            )
