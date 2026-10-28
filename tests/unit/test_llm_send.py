from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
import pytest

from agent.llm import AttemptRecord, LLMClient, LLMError, LLMTransientError
from agent.llm_errors import LLMErrorKind
from agent.model_config import ReasoningSpec, ResolvedModel
from agent.wire import WireRequest, body_sha256, build_wire_request, wire_version


def _model(*, max_retries: int = 0) -> ResolvedModel:
    return ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test-secret-value",
        configured=True,
        model="m",
        context_window=16_000,
        max_tokens=1_000,
        timeout_seconds=10,
        max_retries=max_retries,
        reasoning=ReasoningSpec(),
        pricing=None,
    )


def _client(model: ResolvedModel, handler: Any) -> LLMClient:
    client = LLMClient(model)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _request(model: ResolvedModel, *, stream: bool) -> Any:
    return build_wire_request(
        model,
        [{"role": "user", "content": "hi"}],
        [],
        reasoning_effort=None,
        max_tokens=100,
        stream=stream,
    )


def _ok_json() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def _sse(*chunks: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)


class _ChunkThenDisconnect(httpx.AsyncByteStream):
    """Deliver one SSE delta and then drop the connection mid-stream."""

    def __init__(self) -> None:
        self._sent = False

    async def __aiter__(self):
        if not self._sent:
            self._sent = True
            yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
        raise httpx.ReadError("server disconnected mid-stream")


class _DeltaThenDisconnect(httpx.AsyncByteStream):
    """Deliver one SSE delta from ``delta`` and then fail the stream with ``error``."""

    def __init__(self, delta: dict[str, Any], error: httpx.TransportError) -> None:
        self._delta = delta
        self._error = error
        self._sent = False

    async def __aiter__(self):
        if not self._sent:
            self._sent = True
            yield _sse({"choices": [{"delta": self._delta}]})
        raise self._error


def test_send_sends_exact_body_and_hashes_it() -> None:
    sent: list[bytes] = []
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        assert request.headers["content-type"] == "application/json"
        return httpx.Response(200, json=_ok_json())

    model = _model(max_retries=0)
    client = _client(model, handler)
    request = _request(model, stream=False)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    asyncio.run(client.send(request, on_attempt=on_attempt))

    assert sent == [request.body]
    assert body_sha256(request) == hashlib.sha256(request.body).hexdigest()
    assert records[0].status == "started"
    assert records[0].payload_bytes == len(request.body)
    assert records[0].payload_sha256 == hashlib.sha256(request.body).hexdigest()
    assert records[1].status == "completed"
    assert records[1].prompt_tokens == 7
    assert records[1].completion_tokens == 3


def test_send_retries_503_twice_with_attempt_pairs(monkeypatch: Any) -> None:
    statuses = iter([503, 503, 200])
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json=_ok_json())
        return httpx.Response(status, text="busy")

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=2)
    client = _client(model, handler)
    request = _request(model, stream=False)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    asyncio.run(client.send(request, on_attempt=on_attempt))

    assert [record.attempt for record in records] == [0, 0, 1, 1, 2, 2]
    assert [record.status for record in records] == [
        "started",
        "error",
        "started",
        "error",
        "started",
        "completed",
    ]
    assert records[1].error_kind == LLMErrorKind.server
    assert records[3].error_kind == LLMErrorKind.server
    assert records[5].prompt_tokens == 7


def test_send_error_400_records_error_kind() -> None:
    body = {"error": {"message": "invalid tool schema"}}
    records: list[AttemptRecord] = []
    client = _client(_model(max_retries=0), lambda request: httpx.Response(400, json=body))
    request = _request(client.model, stream=False)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    with pytest.raises(LLMError) as caught:
        asyncio.run(client.send(request, on_attempt=on_attempt))

    assert caught.value.kind == LLMErrorKind.bad_request
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].attempt == 0
    assert records[1].error_kind == LLMErrorKind.bad_request


def test_stream_without_on_delta_raises_value_error() -> None:
    model = _model(max_retries=0)
    client = _client(model, lambda request: httpx.Response(200, json=_ok_json()))
    request = _request(model, stream=True)

    with pytest.raises(ValueError):
        asyncio.run(client.send(request))


def test_send_stream_sends_exact_body(monkeypatch: Any) -> None:
    sent: list[bytes] = []
    records: list[AttemptRecord] = []
    chunks = [{"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}]
    sse = _sse(*chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(200, content=sse)

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=2)
    client = _client(model, handler)
    request = _request(model, stream=True)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    delivered: list[str] = []

    async def on_delta(kind: str, text: str) -> None:
        delivered.append(f"{kind}:{text}")

    response = asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert sent == [request.body]
    assert delivered == ["content:hi"]
    assert response.content == "hi"
    assert [record.status for record in records] == ["started", "completed"]
    assert records[1].payload_sha256 == body_sha256(request)


def test_chat_still_works() -> None:
    client = _client(_model(max_retries=0), lambda request: httpx.Response(200, json=_ok_json()))
    response = asyncio.run(client.chat([{"role": "user", "content": "x"}], []))
    assert response.content == "hi"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 3


def test_chat_propagates_callback_error_when_successful() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["content-type"] == "application/json"
        return httpx.Response(200, json=_ok_json())

    client = _client(_model(max_retries=0), handler)
    request = _request(client.model, stream=False)

    async def failing(record: AttemptRecord) -> None:
        if record.status == "completed":
            raise RuntimeError("audit broke")

    with pytest.raises(RuntimeError, match="audit broke"):
        asyncio.run(client.send(request, on_attempt=failing))

    assert calls == 1


@pytest.mark.parametrize("stream", [False, True])
def test_audit_error_on_error_record_surfaces_after_successful_retry(
    stream: bool, monkeypatch: Any
) -> None:
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 503:
            return httpx.Response(503, text="busy")
        if stream:
            return httpx.Response(
                200,
                content=_sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
            )
        return httpx.Response(200, json=_ok_json())

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=stream)
    records: list[AttemptRecord] = []

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)
        if record.status == "error":
            raise RuntimeError("audit write failed")

    async def on_delta(kind: str, text: str) -> None:
        return None

    with pytest.raises(RuntimeError, match="audit write failed"):
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert [record.status for record in records] == [
        "started",
        "error",
        "started",
        "completed",
    ]
    assert records[1].error_kind == LLMErrorKind.server


@pytest.mark.parametrize("stream", [False, True])
def test_llm_error_keeps_priority_over_audit_error(stream: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad"}})

    model = _model(max_retries=0)
    client = _client(model, handler)
    request = _request(model, stream=stream)
    records: list[AttemptRecord] = []

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)
        if record.status == "error":
            raise RuntimeError("audit write failed")

    async def on_delta(kind: str, text: str) -> None:
        return None

    with pytest.raises(LLMError) as caught:
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert caught.value.kind == LLMErrorKind.bad_request
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.bad_request


def test_invalid_choices_gets_terminal_error_record() -> None:
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _client(_model(max_retries=0), handler)
    request = _request(client.model, stream=False)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    with pytest.raises(LLMError) as caught:
        asyncio.run(client.send(request, on_attempt=on_attempt))

    assert [record.status for record in records] == ["started", "error"]
    assert caught.value.kind == LLMErrorKind.invalid_response
    assert records[1].error_kind == LLMErrorKind.invalid_response


def test_remote_protocol_error_exhausts_as_network(monkeypatch: Any) -> None:
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected")

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    client = _client(_model(max_retries=0), handler)
    request = _request(client.model, stream=False)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    with pytest.raises(LLMTransientError) as caught:
        asyncio.run(client.send(request, on_attempt=on_attempt))

    assert caught.value.kind == LLMErrorKind.network
    assert caught.value.started is False
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.network


def test_remote_protocol_error_retries_in_complete(monkeypatch: Any) -> None:
    calls = 0
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("server disconnected")
        return httpx.Response(200, json=_ok_json())

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=False)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    response = asyncio.run(client.send(request, on_attempt=on_attempt))

    assert calls == 2
    assert response.content == "hi"
    assert [record.status for record in records] == [
        "started",
        "error",
        "started",
        "completed",
    ]
    assert records[1].error_kind == LLMErrorKind.network


def test_remote_protocol_error_retries_before_first_delta(monkeypatch: Any) -> None:
    calls = 0
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("server disconnected")
        return httpx.Response(
            200,
            content=_sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
        )

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)
    delivered: list[str] = []

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        delivered.append(f"{kind}:{text}")

    response = asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert calls == 2
    assert delivered == ["content:hi"]
    assert response.content == "hi"
    assert [record.status for record in records] == [
        "started",
        "error",
        "started",
        "completed",
    ]
    assert records[1].error_kind == LLMErrorKind.network


def test_remote_protocol_error_no_retry_after_delta() -> None:
    calls = 0
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=_DeltaThenDisconnect(
                {"content": "hi"}, httpx.RemoteProtocolError("connection reset")
            ),
        )

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        return None

    with pytest.raises(LLMError) as caught:
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert caught.value.started is True
    assert caught.value.kind == LLMErrorKind.network
    assert calls == 1
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.network


def test_remote_protocol_error_from_on_delta_is_callback_not_transport() -> None:
    calls = 0
    records: list[AttemptRecord] = []
    delta_error = httpx.RemoteProtocolError("callback downstream failed")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "hi"}}]}))

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        raise delta_error

    with pytest.raises(httpx.RemoteProtocolError) as caught:
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert caught.value is delta_error
    assert calls == 1
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.unknown


def test_on_delta_error_gets_terminal_error_record() -> None:
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "hi"}}]}))

    client = _client(_model(max_retries=1), handler)
    request = _request(client.model, stream=True)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        raise RuntimeError("delta broke")

    with pytest.raises(RuntimeError, match="delta broke"):
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.unknown


@pytest.mark.parametrize(
    "delta_error",
    [
        httpx.ReadTimeout("on_delta downstream timed out"),
        httpx.ReadError("on_delta downstream read failed"),
    ],
)
def test_on_delta_httpx_error_is_callback_not_transport(
    delta_error: httpx.TransportError,
) -> None:
    calls = 0
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "hi"}}]}))

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        raise delta_error

    with pytest.raises(type(delta_error)) as caught:
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert caught.value is delta_error
    assert calls == 1
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.unknown


def test_stream_retries_network_failure_before_first_delta(monkeypatch: Any) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("server unreachable")
        return httpx.Response(
            200,
            content=_sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
        )

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)
    records: list[AttemptRecord] = []
    delivered: list[str] = []

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        delivered.append(f"{kind}:{text}")

    response = asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert calls == 2
    assert delivered == ["content:hi"]
    assert response.content == "hi"
    assert [record.status for record in records] == [
        "started",
        "error",
        "started",
        "completed",
    ]
    assert records[1].error_kind == LLMErrorKind.timeout


def test_stream_does_not_retry_after_delta_and_marks_started() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=_ChunkThenDisconnect())

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)
    records: list[AttemptRecord] = []

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        return None

    with pytest.raises(LLMError) as caught:
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert caught.value.started is True
    assert caught.value.kind == LLMErrorKind.network
    assert calls == 1
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.network


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning_content": "thinking"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        },
    ],
)
def test_stream_no_retry_after_substantive_delta_with_audit(delta: dict[str, Any]) -> None:
    calls = 0
    records: list[AttemptRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, content=_DeltaThenDisconnect(delta, httpx.ReadError("server went away"))
        )

    model = _model(max_retries=1)
    client = _client(model, handler)
    request = _request(model, stream=True)

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        return None

    with pytest.raises(LLMError) as caught:
        asyncio.run(client.send(request, on_delta=on_delta, on_attempt=on_attempt))

    assert caught.value.started is True
    assert caught.value.kind == LLMErrorKind.network
    assert calls == 1
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == LLMErrorKind.network


def test_cancel_during_complete_emits_terminal_record() -> None:
    records: list[AttemptRecord] = []
    calls: list[int] = [0]

    async def scenario() -> None:
        in_flight = asyncio.Event()
        release = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            calls[0] += 1
            in_flight.set()
            await release.wait()
            return httpx.Response(200, json=_ok_json())

        model = _model(max_retries=1)
        client = _client(model, handler)
        request = _request(model, stream=False)

        async def on_attempt(record: AttemptRecord) -> None:
            records.append(record)

        task = asyncio.create_task(client.send(request, on_attempt=on_attempt))
        await in_flight.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert calls == [1]
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == "cancelled"


def test_cancel_during_stream_emits_terminal_record() -> None:
    records: list[AttemptRecord] = []
    calls: list[int] = [0]

    async def scenario() -> None:
        in_flight = asyncio.Event()
        release = asyncio.Event()

        class _Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b": keepalive\n\n"
                in_flight.set()
                await release.wait()

        def handler(request: httpx.Request) -> httpx.Response:
            calls[0] += 1
            return httpx.Response(200, content=_Body())

        model = _model(max_retries=1)
        client = _client(model, handler)
        request = _request(model, stream=True)

        async def on_attempt(record: AttemptRecord) -> None:
            records.append(record)

        async def on_delta(kind: str, text: str) -> None:
            return None

        task = asyncio.create_task(client.send(request, on_delta=on_delta, on_attempt=on_attempt))
        await in_flight.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert calls == [1]
    assert [record.status for record in records] == ["started", "error"]
    assert records[1].error_kind == "cancelled"


@pytest.mark.parametrize("stream", [False, True])
def test_send_preserves_noncanonical_body_exactly(stream: bool, monkeypatch: Any) -> None:
    sent: list[bytes] = []
    records: list[AttemptRecord] = []
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        status = next(statuses)
        if status == 503:
            return httpx.Response(503, text="busy")
        if stream:
            return httpx.Response(
                200,
                content=_sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
            )
        return httpx.Response(200, json=_ok_json())

    async def instant_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("agent.llm.asyncio.sleep", instant_sleep)

    model = _model(max_retries=1)
    client = _client(model, handler)
    payload: dict[str, Any] = {
        "model": model.model,
        "messages": [{"role": "user", "content": "hi — привет"}],
        "max_tokens": 100,
        "stream": stream,
    }

    body = json.dumps(payload, ensure_ascii=False, separators=(", ", ": ")).encode("utf-8")
    assert (
        body
        != build_wire_request(
            model,
            [{"role": "user", "content": "hi — привет"}],
            [],
            reasoning_effort=None,
            max_tokens=100,
            stream=stream,
        ).body
    )
    request = WireRequest(payload=payload, body=body, wire_version=wire_version(model))

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    async def on_delta(kind: str, text: str) -> None:
        return None

    asyncio.run(client.send(request, on_delta=on_delta if stream else None, on_attempt=on_attempt))

    assert len(sent) == 2
    expected_hash = hashlib.sha256(request.body).hexdigest()
    for chunk in sent:
        assert chunk == request.body
        assert len(chunk) == len(request.body)
        assert hashlib.sha256(chunk).hexdigest() == expected_hash
        assert hashlib.sha256(chunk).hexdigest() == body_sha256(request)

    assert [record.status for record in records] == ["started", "error", "started", "completed"]
    for record in records:
        assert record.payload_bytes == len(request.body)
        assert record.payload_sha256 == expected_hash
        assert record.payload_sha256 == body_sha256(request)


def test_concurrent_sends_keep_separate_audit_streams() -> None:
    calls: list[int] = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(200, json=_ok_json())

    model = _model(max_retries=0)
    client = _client(model, handler)
    request = _request(model, stream=False)

    async def run() -> tuple[Any, list[AttemptRecord]]:
        records: list[AttemptRecord] = []

        async def on_attempt(record: AttemptRecord) -> None:
            records.append(record)

        response = await client.send(request, on_attempt=on_attempt)
        return response, records

    async def scenario() -> tuple[Any, list[AttemptRecord], Any, list[AttemptRecord]]:
        (first, first_records), (second, second_records) = await asyncio.gather(run(), run())
        return first, first_records, second, second_records

    first, first_records, second, second_records = asyncio.run(scenario())

    assert first.content == "hi"
    assert second.content == "hi"
    assert calls == [2]
    assert [record.status for record in first_records] == ["started", "completed"]
    assert [record.status for record in second_records] == ["started", "completed"]
    assert [record.attempt for record in first_records] == [0, 0]
    assert [record.attempt for record in second_records] == [0, 0]
