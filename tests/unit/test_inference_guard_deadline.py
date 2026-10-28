"""Deterministic deadline/idle/cancellation tests with explicit clock barriers."""

import asyncio
from dataclasses import replace

import httpx
import pytest

from agent.llm import LLMError
from tests.inference_guard_helpers import guard_tasks as guard_tasks
from tests.unit.test_llm_send import _client, _model, _request, _sse

pytestmark = pytest.mark.usefixtures("guard_tasks")


class Clock:
    def __init__(self):
        self.value = 0.0
        self.waiters = []
        self.armed = asyncio.Queue()

    def now(self):
        return self.value

    async def until(self, deadline):
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((deadline, future))
        self.armed.put_nowait(deadline)
        if self.value >= deadline:
            future.set_result(None)
        await future

    def advance(self, value):
        self.value = value
        for deadline, future in self.waiters:
            if deadline <= value and not future.done():
                future.set_result(None)


async def test_total_covers_prepare_and_ignores_suppressed_cancel():
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        clock = Clock()
        guard = GuardContext(_model(), clock=clock)
        started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def prepare():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return "late"

        task = asyncio.create_task(guard.run(prepare()))
        await started.wait()
        assert await clock.armed.get() == 600
        clock.advance(600)
        await cancelled.wait()
        # Work that suppresses cancellation must not hold the observer.
        with pytest.raises(LLMError) as caught:
            await task
        assert caught.value.kind.value == "llm_inference_timeout"
        assert guard.closed
        release.set()


async def test_model_change_only_narrows_shared_deadline_between_retries():
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        clock = Clock()
        guard = GuardContext(_model(), clock=clock)
        between, release = asyncio.Event(), asyncio.Event()
        attempts = []

        async def work():
            attempts.append(1)
            guard.select(replace(_model(), inference_timeout_seconds=200))
            between.set()
            await release.wait()
            guard.ensure_open()
            attempts.append(2)

        task = asyncio.create_task(guard.run(work()))
        await between.wait()
        while await clock.armed.get() != 200:
            pass
        clock.advance(200)
        with pytest.raises(LLMError):
            await task
        assert attempts == [1]
        assert guard.deadline == 200
        guard.select(replace(_model(), inference_timeout_seconds=900))
        assert guard.deadline == 200


@pytest.mark.parametrize(
    "raw,progress",
    [
        ({"content": "x"}, True),
        ({"reasoning_content": "x"}, True),
        ({"content": " \n"}, False),
        ({"role": "assistant"}, False),
        ({"tool_calls": []}, False),
        ({"function_call": {"name": "read"}}, True),
        ({"function_call": {"arguments": "{"}}, True),
        ({"functions_state_id": "state"}, False),
    ],
)
async def test_semantic_idle_on_original_data(raw, progress):
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        clock = Clock()
        model = replace(_model(), dialect="functions")
        guard = GuardContext(model, clock=clock)
        read, release = asyncio.Event(), asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                read.set()
                await release.wait()
                yield _sse({"choices": [{"delta": raw}]})
                await asyncio.Event().wait()

        client = _client(model, lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            pass

        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(
                guard.run(client.send(_request(model, stream=True), on_delta=delta))
            )
            await read.wait()
            while await clock.armed.get() != 120:
                pass
            clock.advance(100)
            release.set()
            # Clock barrier after processing the delta: progress re-arms idle at 220.
            if progress:
                while await clock.armed.get() != 220:
                    pass
            clock.advance(120 if not progress else 220)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_semantic_idle"
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


async def test_nonstream_silence_150_seconds_and_explicit_read_timeout():
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        clock = Clock()
        guard = GuardContext(_model(), clock=clock)
        arrived, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            assert request.extensions["timeout"]["read"] == 600
            arrived.set()
            await release.wait()
            return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

        client = _client(_model(), handler)
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(guard.run(client.send(_request(client.model, stream=False))))
            await arrived.wait()
            clock.advance(150)
            release.set()
            assert (await task).content == "answer"
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


async def test_cleanup_uses_one_two_second_boundary_and_never_joins_pending():
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        clock = Clock()
        guard = GuardContext(_model(), clock=clock)
        guard.close(asyncio.CancelledError())
        started, release = asyncio.Event(), asyncio.Event()

        async def stuck():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return "late"

        first = asyncio.create_task(guard.cleanup(stuck()))
        await started.wait()
        assert await clock.armed.get() == 2
        clock.advance(2)
        assert await first is None
        started.clear()
        second = asyncio.create_task(guard.cleanup(stuck()))
        await started.wait()
        assert await clock.armed.get() == 2
        assert await second is None
        assert guard.cleanup_deadline == 2
        release.set()


async def test_manual_stop_keeps_cancelled_error_and_closes_publisher():
    async with asyncio.timeout(5):
        from agent.events import StreamPublisher, broker
        from agent.inference_guard import GuardContext

        guard = GuardContext(_model())
        publisher = StreamPublisher("s")
        guard.on_close(publisher.close)
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            await publisher.add("content", "pending")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            await publisher.add("content", "late")
            await publisher.flush()

        async with broker.subscribe() as queue:
            task = asyncio.create_task(guard.run(work()))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert guard.closed and publisher.closed
            release.set()
            assert queue.empty()


@pytest.mark.parametrize(
    "point,stream",
    [
        ("auth", False),
        ("auth", True),
        ("backoff", False),
        ("read", True),
        ("callback", True),
        ("close", True),
    ],
)
async def test_stalled_transport_boundaries(point, stream, monkeypatch):
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        clock = Clock()
        model = _model(max_retries=3)
        guard = GuardContext(model, clock=clock)
        entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                if point == "read":
                    entered.set()
                    await release.wait()
                yield _sse({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]})

            async def aclose(self):
                if point == "close":
                    entered.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        await release.wait()
                    exited.set()

        def handler(request):
            calls.append(request)
            return (
                httpx.Response(503) if point == "backoff" else httpx.Response(200, stream=Stream())
            )

        client = _client(model, handler)
        if point == "auth":

            async def headers():
                entered.set()
                await release.wait()
                return {}, None

            monkeypatch.setattr(client, "_headers", headers)
        if point == "backoff":
            real_sleep = asyncio.sleep

            async def backoff(seconds):
                if seconds:
                    entered.set()
                    await release.wait()
                else:
                    await real_sleep(0)

            monkeypatch.setattr("agent.llm.asyncio.sleep", backoff)

        async def delta(kind, text):
            if point == "callback":
                entered.set()
                await release.wait()

        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(
                guard.run(
                    client.send(_request(model, stream=stream), on_delta=delta if stream else None)
                )
            )
            await entered.wait()
            clock.advance(120 if stream else 600)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == (
                "llm_semantic_idle" if stream else "llm_inference_timeout"
            )
            assert len(calls) <= 1
            release.set()
            if point == "close":
                await exited.wait()
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


async def test_meaningful_forever_stream_is_stopped_by_total():
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        clock = Clock()
        guard = GuardContext(_model(), clock=clock)
        input_queue, published = asyncio.Queue(), asyncio.Queue()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    text = await input_queue.get()
                    yield _sse({"choices": [{"delta": {"content": text}}]})

        client = _client(_model(), lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            published.put_nowait(text)

        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(
                guard.run(client.send(_request(client.model, stream=True), on_delta=delta))
            )
            while await clock.armed.get() != 120:
                pass
            for now in (100, 200, 300, 400, 500):
                clock.advance(now)
                input_queue.put_nowait(str(now))
                assert await published.get() == str(now)
            clock.advance(600)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_inference_timeout"
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


@pytest.mark.parametrize(
    "second,extends",
    [
        ({"function_call": {"name": "read"}}, False),
        ({"function_call": {"arguments": "{"}}, True),
        ({"functions_state_id": "state"}, False),
        ({"role": "assistant"}, False),
        ({"content": " \n"}, False),
    ],
)
async def test_repeated_metadata_does_not_extend_but_argument_characters_do(second, extends):
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        model = replace(_model(), dialect="functions")
        clock = Clock()
        guard = GuardContext(model, clock=clock)
        frames, processed = asyncio.Queue(), asyncio.Queue()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    raw = await frames.get()
                    yield _sse({"choices": [{"delta": raw}]})
                    processed.put_nowait(True)

        client = _client(model, lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            pass

        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(
                guard.run(client.send(_request(model, stream=True), on_delta=delta))
            )
            while await clock.armed.get() != 120:
                pass
            clock.advance(100)
            frames.put_nowait({"function_call": {"name": "read", "arguments": "{"}})
            await processed.get()
            clock.advance(200)
            frames.put_nowait(second)
            await processed.get()
            clock.advance(320 if extends else 220)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_semantic_idle"
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


async def test_session_deadline_wins_and_result_at_boundary_is_rejected():
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext
        from agent.session_budget_store import BudgetExceeded

        clock = Clock()
        guard = GuardContext(_model(), clock=clock, session_deadline=30)
        entered = asyncio.Event()

        async def work():
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(guard.run(work()))
        await entered.wait()
        clock.advance(30)
        with pytest.raises(BudgetExceeded) as caught:
            await task
        assert caught.value.reason == "deadline_exceeded"
        guard = GuardContext(_model(), clock=clock)

        async def late_result():
            clock.advance(630)
            return "late"

        with pytest.raises(LLMError) as caught_llm:
            await guard.run(late_result())
        assert caught_llm.value.kind.value == "llm_inference_timeout"


async def test_auth_refresh_uses_original_total(monkeypatch):
    async with asyncio.timeout(5):
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        clock = Clock()
        guard = GuardContext(_model(), clock=clock)
        first, release, refreshing = asyncio.Event(), asyncio.Event(), asyncio.Event()
        headers_calls = []

        async def handler(request):
            first.set()
            await release.wait()
            return httpx.Response(401, text="expired")

        client = _client(_model(), handler)

        class Auth:
            refreshable = True

            def invalidate(self, token):
                assert token == "old"

        monkeypatch.setattr(client, "_auth_provider", lambda: Auth())

        async def headers():
            headers_calls.append(1)
            if len(headers_calls) > 1:
                refreshing.set()
                await asyncio.Event().wait()
            return {}, "old"

        monkeypatch.setattr(client, "_headers", headers)
        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(guard.run(client.send(_request(client.model, stream=False))))
            await first.wait()
            clock.advance(400)
            release.set()
            await refreshing.wait()
            clock.advance(600)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_inference_timeout"
            assert len(headers_calls) == 2
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


async def test_outer_deadline_interrupts_parsing_a_single_transport_chunk(monkeypatch):
    async with asyncio.timeout(5):
        from agent import sse
        from agent.inference_guard import CURRENT_GUARD, GuardContext

        clock = Clock()
        model = replace(_model(), inference_timeout_seconds=200, repetition_threshold_chars=None)
        guard = GuardContext(model, clock=clock)
        reading, release, published = asyncio.Event(), asyncio.Event(), asyncio.Event()
        decoded = []
        original = sse.ChatStreamAccumulator.add_payload

        def add_payload(self, payload):
            decoded.append(1)
            return original(self, payload)

        monkeypatch.setattr(sse.ChatStreamAccumulator, "add_payload", add_payload)
        content = _sse({"choices": [{"delta": {"content": "answer"}}]})
        content += _sse(*[{"choices": [{"delta": {"role": "assistant"}}]}] * 10000)
        content += _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                reading.set()
                await release.wait()
                yield content

        client = _client(model, lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            published.set()

        token = CURRENT_GUARD.set(guard)
        try:
            task = asyncio.create_task(
                guard.run(client.send(_request(model, stream=True), on_delta=delta))
            )
            await reading.wait()
            clock.advance(100)
            release.set()
            await published.wait()
            clock.advance(200)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_inference_timeout"
            assert len(decoded) < 10002
        finally:
            CURRENT_GUARD.reset(token)
            await client.aclose()


@pytest.mark.parametrize("failure", ["empty", "repetition", "invalid"])
async def test_guard_closes_before_stalled_response_exit(failure):
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        clock = Clock()
        model = replace(_model(), repetition_threshold_chars=4096)
        guard = GuardContext(model, clock=clock)
        closing, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        content = {"empty": " ", "repetition": "x" * 4096, "invalid": "answer"}[failure]
        reason = 17 if failure == "invalid" else "stop"

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _sse({"choices": [{"delta": {"content": content}, "finish_reason": reason}]})

            async def aclose(self):
                closing.set()
                try:
                    await release.wait()
                finally:
                    exited.set()

        client = _client(model, lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            pass

        task = asyncio.create_task(
            guard.run(client.send(_request(model, stream=True), on_delta=delta))
        )
        try:
            await closing.wait()
            assert guard.closed, "response exit started before the guard closed"
            assert guard.cleanup_deadline == 2
            with pytest.raises(LLMError) as caught:
                await task
            assert (
                caught.value.kind.value
                == {
                    "empty": "llm_empty_response",
                    "repetition": "llm_repetition",
                    "invalid": "llm_invalid_response",
                }[failure]
            )
            clock.advance(2)
            await exited.wait()
        finally:
            release.set()
            task.cancel()
            await client.aclose()


async def test_cleanup_failure_cannot_replace_stream_guard(monkeypatch):
    async with asyncio.timeout(5):
        from agent import inference_guard as module
        from agent.inference_guard import GuardContext

        logged = asyncio.Event()
        diagnostics = []
        original_warning = module.logger.warning

        def warning(*args, **kwargs):
            diagnostics.append(args[0])
            original_warning(*args, **kwargs)
            logged.set()

        monkeypatch.setattr(module.logger, "warning", warning)
        guard = GuardContext(_model())
        exited = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})

            async def aclose(self):
                exited.set()
                raise RuntimeError("cleanup failed")

        client = _client(_model(), lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            pass

        try:
            with pytest.raises(LLMError) as caught:
                await guard.run(client.send(_request(client.model, stream=True), on_delta=delta))
            assert caught.value.kind.value == "llm_empty_response"
            await exited.wait()
            await logged.wait()
            assert diagnostics == ["Inference cleanup failed: %s"]
        finally:
            await client.aclose()


async def test_cancel_suppressing_backoff_cannot_start_retry_auth(monkeypatch):
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        clock = Clock()
        model = _model(max_retries=1)
        guard = GuardContext(model, clock=clock)
        backoff, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        headers = []
        client = _client(model, lambda _: httpx.Response(503))
        real_sleep = asyncio.sleep

        async def sleep(seconds):
            if not seconds:
                return await real_sleep(0)
            backoff.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

        async def auth():
            headers.append(1)
            return {}, None

        async def work():
            try:
                await client.send(_request(model, stream=False))
            finally:
                exited.set()

        monkeypatch.setattr(client, "_headers", auth)
        monkeypatch.setattr("agent.llm.asyncio.sleep", sleep)
        task = asyncio.create_task(guard.run(work()))
        try:
            await backoff.wait()
            clock.advance(600)
            with pytest.raises(LLMError):
                await task
            release.set()
            await exited.wait()
            assert headers == [1], "retry auth started after the logical deadline"
        finally:
            release.set()
            task.cancel()
            await client.aclose()


async def test_nonstream_total_precedes_the_six_hundred_second_read_limit():
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        model = replace(_model(), inference_timeout_seconds=30, semantic_idle_timeout_seconds=20)
        clock = Clock()
        guard = GuardContext(model, clock=clock)
        arrived = asyncio.Event()

        async def handler(request):
            assert request.extensions["timeout"]["read"] == 600
            arrived.set()
            await asyncio.Event().wait()

        client = _client(model, handler)
        task = asyncio.create_task(guard.run(client.send(_request(model, stream=False))))
        try:
            await arrived.wait()
            clock.advance(30)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_inference_timeout"
        finally:
            task.cancel()
            await client.aclose()


async def test_heartbeat_comments_and_usage_do_not_extend_idle():
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext

        model, clock = _model(), Clock()
        guard = GuardContext(model, clock=clock)
        frames, processed = asyncio.Queue(), asyncio.Queue()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    yield await frames.get()
                    processed.put_nowait(True)

        client = _client(model, lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            pytest.fail("metadata was published as text")

        task = asyncio.create_task(
            guard.run(client.send(_request(model, stream=True), on_delta=delta))
        )
        try:
            while await clock.armed.get() != 120:
                pass
            clock.advance(100)
            frames.put_nowait(
                b": heartbeat\n\n" + _sse({"choices": [], "usage": {"prompt_tokens": 7}})
            )
            await processed.get()
            clock.advance(120)
            with pytest.raises(LLMError) as caught:
                await task
            assert caught.value.kind.value == "llm_semantic_idle"
        finally:
            task.cancel()
            await client.aclose()


async def test_publishers_capture_run_and_monotonic_stream_identity():
    async with asyncio.timeout(5):
        from agent.events import CURRENT_STREAMS, StreamPublisher, StreamRun, broker
        from agent.session_fence import CURRENT_FENCE, RunFence

        run = StreamRun(7)
        token = CURRENT_STREAMS.set(run)
        fence_token = CURRENT_FENCE.set(RunFence("s", 7, 1, "owner"))
        try:
            first, second = StreamPublisher("s"), StreamPublisher("s")
            assert (first.stream_no, second.stream_no) == (1, 2)
            assert first.stream_id != second.stream_id
            CURRENT_FENCE.set(RunFence("s", 8, 2, "next"))
            async with broker.subscribe() as queue:
                await first.add("content", "accepted")
                await first.flush()
                await first.reset()
                delta, reset = queue.get_nowait(), queue.get_nowait()
                assert delta["payload"] == {**first.identity, "seq": 1, "delta": "accepted"}
                assert reset["payload"] == {**first.identity, "seq": 2}
                assert first.generation == 7 and run.number == 2
                await first.add("content", "late")
                await first.flush()
                assert queue.empty()
            CURRENT_STREAMS.set(StreamRun(8))
            next_run = StreamPublisher("s")
            assert (next_run.generation, next_run.stream_no) == (8, 1)
        finally:
            CURRENT_STREAMS.reset(token)
            CURRENT_FENCE.reset(fence_token)


@pytest.mark.parametrize("boundary", ["narrowed_on_result", "session_zero"])
async def test_observer_uses_current_boundary_when_worker_finishes_in_same_turn(boundary):
    async with asyncio.timeout(5):
        from agent.inference_guard import GuardContext
        from agent.session_budget_store import BudgetExceeded

        clock = Clock()
        guard = GuardContext(
            _model(), clock=clock, session_deadline=0 if boundary == "session_zero" else None
        )

        async def work():
            if boundary == "narrowed_on_result":
                clock.advance(250)
                guard.select(replace(_model(), inference_timeout_seconds=200))
            return "late result"

        with pytest.raises(LLMError if boundary == "narrowed_on_result" else BudgetExceeded):
            await guard.run(work())
