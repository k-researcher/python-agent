"""Tests for the async agent API client (agent.cli_client)."""

from __future__ import annotations

import json as _json_mod
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from httpx import Request, Response

from agent.cli_client.api import AgentApi, ApiError, ClientConfig

TOKEN = "top-secret-token"


class _StopLoop(BaseException):
    """Break out of the event reconnect loop in tests."""


def _client(
    handler: Callable[[Request], Response],
    *,
    token: str = TOKEN,
    base_url: str = "http://127.0.0.1:8080",
) -> AgentApi:
    transport = httpx.MockTransport(handler)
    return AgentApi(ClientConfig(base_url, token), transport=transport)


def _json(request: Request, payload: object, status: int = 200) -> Response:
    return Response(status, json=payload, request=request)


def _body(request: Request) -> dict:
    return _json_mod.loads(request.read())


def _assert_token(request: Request) -> None:
    assert request.headers.get("X-Agent-Token") == TOKEN


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# --- token header on every request ----------------------------------------


def test_token_header_present_on_requests() -> None:
    seen: list[str] = []

    def handler(request: Request) -> Response:
        _assert_token(request)
        seen.append(request.method)
        if request.method == "GET":
            return _json(request, [])
        return _json(
            request, {"id": "p9", "name": "x", "root_path": "/tmp/a", "created_at": "t"}, status=201
        )

    api = _client(handler)

    async def run() -> None:
        await api.projects()
        await api.ensure_project("/tmp/a", name="x")
        await api.aclose()

    _run(run())
    assert "GET" in seen and "POST" in seen


# --- ensure_project -------------------------------------------------------


def test_ensure_project_returns_matching_existing() -> None:
    calls: list[str] = []

    def handler(request: Request) -> Response:
        _assert_token(request)
        calls.append(request.method)
        if request.method == "GET":
            return _json(
                request, [{"id": "p1", "name": "one", "root_path": "/tmp/a", "created_at": "t"}]
            )
        raise AssertionError("must not POST when a match exists")

    api = _client(handler)
    result = _run(api.ensure_project("/tmp/a", name="one"))
    assert result["id"] == "p1"
    assert calls == ["GET"]


def test_ensure_project_creates_new() -> None:
    created: list[dict] = []

    def handler(request: Request) -> Response:
        _assert_token(request)
        if request.method == "GET":
            return _json(request, [])
        assert request.method == "POST"
        assert request.url.path == "/api/projects"
        created.append(_body(request))
        return _json(
            request,
            {"id": "p9", "name": "created", "root_path": "/tmp/a", "created_at": "t"},
            status=201,
        )

    api = _client(handler)
    result = _run(api.ensure_project("/tmp/a", name="created"))
    assert result["id"] == "p9"
    # The client sends the resolved path (on macOS /tmp is a link to /private/tmp).
    assert created[0]["root_path"] == str(Path("/tmp/a").resolve())
    assert created[0]["name"] == "created"


def test_ensure_project_defaults_name_to_basename() -> None:
    created: list[dict] = []

    def handler(request: Request) -> Response:
        if request.method == "POST":
            created.append(_body(request))
            return _json(
                request,
                {"id": "p9", "name": "", "root_path": "/tmp/p", "created_at": "t"},
                status=201,
            )
        return _json(request, [])

    api = _client(handler)
    _run(api.ensure_project("/tmp/my-project"))
    assert created[0]["name"] == "my-project"


def test_ensure_project_resolves_absolute_path_and_symlinks() -> None:
    import tempfile as _tf

    root = Path(_tf.mkdtemp())
    target = root / "real-dir"
    target.mkdir()
    link = root / "link-to-dir"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not available")
    registered = str(target)

    seen: list[dict] = []

    def handler(request: Request) -> Response:
        if request.method == "GET":
            return _json(
                request, [{"id": "p1", "name": "x", "root_path": registered, "created_at": "t"}]
            )
        seen.append(_body(request))
        return _json(request, {}, status=201)

    api = _client(handler)
    # Passing a symlink must still match the real registered path.
    result = _run(api.ensure_project(str(link)))
    assert result["id"] == "p1"
    assert seen == []


def test_ensure_project_rejects_empty_path() -> None:
    api = _client(lambda request: _json(request, []))
    with pytest.raises(ApiError) as exc_info:
        _run(api.ensure_project("   "))
    assert "must not be empty" in exc_info.value.detail


# --- ApiError -------------------------------------------------------------


def test_api_error_carries_detail() -> None:
    def handler(request: Request) -> Response:
        return _json(request, {"detail": "Project directory is already registered"}, status=409)

    api = _client(handler)
    with pytest.raises(ApiError) as exc_info:
        _run(api.ensure_project("/tmp/some-path", name="x"))
    assert exc_info.value.status == 409
    assert exc_info.value.detail == "Project directory is already registered"
    assert "Project directory is already registered" in str(exc_info.value)


def test_api_error_uses_body_text_when_no_json_detail() -> None:
    def handler(request: Request) -> Response:
        return Response(500, text="boom", request=request)

    api = _client(handler)
    with pytest.raises(ApiError) as exc_info:
        _run(api.projects())
    assert exc_info.value.status == 500
    assert exc_info.value.detail == "boom"


def test_transport_error_wrapped_as_api_error() -> None:
    class Boom(httpx.HTTPError):
        pass

    def handler(request: Request) -> Response:
        raise Boom("network down")

    api = _client(handler)
    with pytest.raises(ApiError) as exc_info:
        _run(api.projects())
    assert "network down" in exc_info.value.detail


def test_invalid_json_response_raises_api_error() -> None:
    def handler(request: Request) -> Response:
        return Response(200, text="not json", request=request)

    api = _client(handler)
    with pytest.raises(ApiError):
        _run(api.projects())


# --- empty token and secrecy ---------------------------------------------


def test_empty_token_raises_clear_error() -> None:
    api = _client(lambda request: _json(request, []), token="")
    with pytest.raises(ApiError) as exc_info:
        _run(api.projects())
    assert "AGENT_API_TOKEN" in exc_info.value.detail
    assert exc_info.value.status == 401


def test_token_not_in_errors_or_repr() -> None:
    config = ClientConfig("http://127.0.0.1:8080", TOKEN)
    assert TOKEN not in repr(config)

    def handler(request: Request) -> Response:
        return _json(request, {"detail": "Unauthorized"}, status=401)

    api = _client(handler)
    with pytest.raises(ApiError) as exc_info:
        _run(api.projects())
    assert TOKEN not in str(exc_info.value)
    assert TOKEN not in repr(exc_info.value)


# --- request body shapes --------------------------------------------------


def test_create_session_body() -> None:
    captured: dict = {}

    def handler(request: Request) -> Response:
        _assert_token(request)
        captured.update(_body(request))
        return _json(request, {"id": "s1", "project_id": "p1", "created_at": "t"}, status=201)

    api = _client(handler)
    _run(
        api.create_session(
            "p1", "hello", mode="ask", llm_profile="fast", reasoning_effort="high", auto_start=False
        )
    )
    assert captured == {
        "project_id": "p1",
        "prompt": "hello",
        "mode": "ask",
        "llm_profile": "fast",
        "reasoning_effort": "high",
        "auto_start": False,
    }


def test_send_message_body() -> None:
    captured: list[dict] = []

    def handler(request: Request) -> Response:
        captured.append(_body(request))
        return _json(request, {"id": 1, "content": "x", "created_at": "t"}, status=201)

    api = _client(handler)
    _run(api.send_message("s1", "hi", reasoning_effort="low"))
    _run(api.send_message("s2", "hi again"))
    assert captured[0] == {"content": "hi", "reasoning_effort": "low"}
    assert captured[1] == {"content": "hi again"}


def test_resolve_approval_body() -> None:
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured.update(_body(request))
        return _json(request, {"session_id": "s1", "decision": "approve"})

    api = _client(handler)
    _run(api.resolve_approval("a1", "approve", comment="ok"))
    assert captured == {"decision": "approve", "comment": "ok"}


def test_resolve_approval_without_comment() -> None:
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured.update(_body(request))
        return _json(request, {})

    api = _client(handler)
    _run(api.resolve_approval("a1", "reject"))
    assert captured == {"decision": "reject"}


# --- WebSocket events -----------------------------------------------------


async def _collect_n(api: AgentApi, session: str | None, n: int) -> list[dict]:
    """Collect up to ``n`` events, then close the stream to stop the reconnect loop."""
    out: list[dict] = []
    gen = api.events(session)
    try:
        async for event in gen:
            out.append(event)
            if len(out) >= n:
                break
    finally:
        await gen.aclose()
    return out


def test_events_filter_by_session_preserving_order() -> None:
    async def fake_connect(uri: str) -> AsyncIterator[dict]:
        assert uri == "ws://127.0.0.1:8080/ws"
        for raw in [
            {"type": "a", "session_id": "s1", "payload": {}},
            {"type": "b", "session_id": "s2", "payload": {}},
            {"type": "c", "session_id": "s1", "payload": {}},
        ]:
            yield raw

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    events = _run(_collect_n(api, "s1", 2))
    assert [e["type"] for e in events] == ["a", "c"]


def test_events_yields_all_when_no_filter() -> None:
    async def fake_connect(uri: str) -> AsyncIterator[dict]:
        yield {"type": "a", "session_id": None, "payload": {}}
        yield {"type": "b", "session_id": "s2", "payload": {}}

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    events = _run(_collect_n(api, None, 2))
    assert [e["type"] for e in events] == ["a", "b"]


def test_events_handles_json_text_and_malformed() -> None:
    async def fake_connect(uri: str) -> AsyncIterator[object]:
        yield _json_mod.dumps({"type": "a", "session_id": "s1", "payload": {}})
        yield "not :: json"
        yield b'{"type":"b","session_id":"s1","payload":{}}'

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    events = _run(_collect_n(api, "s1", 2))
    assert [e["type"] for e in events] == ["a", "b"]


def test_events_reconnect_with_growing_delay() -> None:
    delays: list[float] = []
    attempts = 0

    async def patched_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_connect(uri: str) -> AsyncIterator[dict]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:

            async def gen() -> AsyncIterator[dict]:
                yield {"type": "d", "session_id": "s1", "payload": {}}
                raise ConnectionError("dropped")

            return gen()

        async def gen() -> AsyncIterator[dict]:
            yield {"type": "ok", "session_id": "s1", "payload": {}}

        return gen()

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    api._sleep = patched_sleep  # type: ignore[assignment]
    events = _run(_collect_n(api, "s1", 2))
    # Attempt 1 dropped after the first event; a single reconnect with 1.0 s.
    assert [e["type"] for e in events] == ["d", "ok"]
    assert delays == [1.0]
    assert attempts == 2


def test_events_reconnect_delay_grows() -> None:
    delays: list[float] = []

    async def patched_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 4:
            raise _StopLoop

    async def always_fail(uri: str) -> AsyncIterator[dict]:

        async def gen() -> AsyncIterator[dict]:
            raise ConnectionError("down")
            yield  # pragma: no cover - makes this an async generator

        return gen()

    api = _client(lambda request: _json(request, []))
    api._ws_connect = always_fail  # type: ignore[assignment]
    api._sleep = patched_sleep  # type: ignore[assignment]

    async def run_and_exit() -> None:
        async for _ in api.events():  # pragma: no cover - never yields
            pass

    with pytest.raises(_StopLoop):
        _run(run_and_exit())
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_events_delay_caps_at_30() -> None:
    seen: list[float] = []

    async def patched_sleep(delay: float) -> None:
        seen.append(delay)
        if len(seen) >= 8:
            raise _StopLoop

    async def always_fail(uri: str) -> AsyncIterator[dict]:

        async def gen() -> AsyncIterator[dict]:
            raise ConnectionError("down")
            yield  # pragma: no cover - makes this an async generator

        return gen()

    api = _client(lambda request: _json(request, []))
    api._ws_connect = always_fail  # type: ignore[assignment]
    api._sleep = patched_sleep  # type: ignore[assignment]

    async def run_and_exit() -> None:
        async for _ in api.events():  # pragma: no cover - never yields
            pass

    with pytest.raises(_StopLoop):
        _run(run_and_exit())
    assert seen == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]


def test_events_delay_resets_after_success() -> None:
    delays: list[float] = []
    attempts = 0

    async def patched_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_connect(uri: str) -> AsyncIterator[dict]:
        nonlocal attempts
        attempts += 1
        if attempts in (1, 3):

            async def gen() -> AsyncIterator[dict]:
                yield {"type": f"e{attempts}", "session_id": "s1", "payload": {}}
                raise ConnectionError("drop")

            return gen()

        async def gen() -> AsyncIterator[dict]:
            yield {"type": "ok", "session_id": "s1", "payload": {}}

        return gen()

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    api._sleep = patched_sleep  # type: ignore[assignment]
    # Attempt 1: e1 + drop (sleep 1.0). Attempt 2: ok (delay resets to 1.0).
    # Attempt 3: e3 + drop (sleep 1.0 again). Attempt 4: ok.
    events = _run(_collect_n(api, "s1", 4))
    assert [e["type"] for e in events] == ["e1", "ok", "e3", "ok"]
    assert delays == [1.0, 1.0]
    assert attempts == 4


def test_events_stops_on_401() -> None:
    async def fake_connect(uri: str) -> AsyncIterator[dict]:
        raise ApiError(401, "Unauthorized")

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    with pytest.raises(ApiError) as exc_info:
        _run(_collect_n(api, None, 1))
    assert exc_info.value.status == 401


def test_events_stops_on_403_from_plain_exception() -> None:
    class Denied(Exception):
        status_code = 403

    async def fake_connect(uri: str) -> AsyncIterator[dict]:
        raise Denied("forbidden")

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]
    with pytest.raises(ApiError) as exc_info:
        _run(_collect_n(api, None, 1))
    assert exc_info.value.status == 403


def test_events_empty_token() -> None:
    api = _client(lambda request: _json(request, []), token="")
    with pytest.raises(ApiError) as exc_info:
        _run(_collect_n(api, None, 1))
    assert "AGENT_API_TOKEN" in exc_info.value.detail


def test_events_cancellation_closes_connection() -> None:
    import asyncio

    closed: list[bool] = []

    async def fake_connect(uri: str) -> AsyncIterator[dict]:

        async def gen() -> AsyncIterator[dict]:
            try:
                yield {"type": "a", "session_id": "s1", "payload": {}}
                await asyncio.sleep(3600)
            finally:
                closed.append(True)

        return gen()

    api = _client(lambda request: _json(request, []))
    api._ws_connect = fake_connect  # type: ignore[assignment]

    async def run() -> None:
        gen = api.events()
        it = gen.__aiter__()
        await it.__anext__()
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert closed


def test_aclose_is_cancellation_safe() -> None:
    import asyncio
    import contextlib

    api = _client(lambda request: _json(request, []))

    async def run() -> None:
        task = asyncio.create_task(api.aclose())
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())


# --- misc -----------------------------------------------------------------


def test_from_env_defaults() -> None:
    import os

    old_url = os.environ.get("AGENT_URL")
    old_token = os.environ.get("AGENT_API_TOKEN")
    try:
        os.environ["AGENT_URL"] = "http://example.com:9999"
        os.environ["AGENT_API_TOKEN"] = "env-token"
        config = ClientConfig.from_env()
        assert config.base_url == "http://example.com:9999"
        assert config.token == "env-token"
        assert "env-token" not in repr(config)
    finally:
        if old_url is None:
            os.environ.pop("AGENT_URL", None)
        else:
            os.environ["AGENT_URL"] = old_url
        if old_token is None:
            os.environ.pop("AGENT_API_TOKEN", None)
        else:
            os.environ["AGENT_API_TOKEN"] = old_token


def test_from_env_explicit_overrides_env() -> None:
    config = ClientConfig.from_env(base_url="http://custom:1", token="t")
    assert config.base_url == "http://custom:1"
    assert config.token == "t"


def test_api_error_repr() -> None:
    err = ApiError(409, "conflict")
    assert repr(err) == "ApiError(status=409, detail='conflict')"


def test_ensure_project_idempotent_no_duplicate_posts() -> None:
    posts = 0

    def handler(request: Request) -> Response:
        nonlocal posts
        if request.method == "GET":
            return _json(
                request, [{"id": "p1", "name": "x", "root_path": "/tmp/a", "created_at": "t"}]
            )
        posts += 1
        return _json(request, {})

    api = _client(handler)
    _run(api.ensure_project("/tmp/a"))
    _run(api.ensure_project("/tmp/a"))
    assert posts == 0


@pytest.mark.parametrize("status", [400, 404, 409, 500])
def test_token_is_redacted_in_server_errors(status: int) -> None:
    api = _client(lambda request: _json(request, {"detail": f"bad {TOKEN} here"}, status))
    with pytest.raises(ApiError) as caught:
        _run(api.sessions())
    assert TOKEN not in caught.value.detail
    assert TOKEN not in str(caught.value)
    assert "***" in caught.value.detail
    assert caught.value.status == status
