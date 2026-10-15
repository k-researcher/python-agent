"""LLM transport with flexible providers: auth, headers, CA bundle and the functions dialect."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent.llm import AttemptRecord, LLMClient, LLMError
from agent.llm_functions_dialect import to_functions_request
from agent.model_config import (
    BearerAuthSpec,
    HeaderKeyAuthSpec,
    OAuthAuthSpec,
    ReasoningSpec,
    ResolvedModel,
)
from agent.provider_auth import AuthProvider, OAuthClientCredentials
from agent.wire import body_sha256, build_wire_request, wire_version

AUTH_KEY = "basic-auth-key-secret"
TOKEN_URL = "https://auth.example.test/api/v2/oauth"


def _model(**changes: Any) -> ResolvedModel:
    model = ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url="https://llm.example.test/v1",
        api_key="",
        configured=True,
        model="m",
        context_window=16_000,
        max_tokens=1_000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
    )
    return replace(model, **changes)


def _oauth_model(**changes: Any) -> ResolvedModel:
    spec = OAuthAuthSpec(
        type="oauth_client_credentials", token_url=TOKEN_URL, key_env="AUTH_KEY", scope="API"
    )
    return _model(auth=spec, auth_secret=AUTH_KEY, **changes)


def _request(model: ResolvedModel, *, stream: bool = False, tools: bool = False) -> Any:
    tool = {"type": "function", "function": {"name": "read_file", "parameters": {}}}
    return build_wire_request(
        model,
        [{"role": "user", "content": "hi"}],
        [tool] if tools else [],
        reasoning_effort=None,
        max_tokens=100,
        stream=stream,
    )


def _ok() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


class TokenServer:
    """Issue tokens "token-1", "token-2", ... and count the requests."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        number = len(self.requests)
        return httpx.Response(
            200, json={"access_token": f"token-{number}", "expires_at": 4_102_444_800_000}
        )


def _client(model: ResolvedModel, llm: Any, tokens: TokenServer | None = None) -> LLMClient:
    client = LLMClient(model)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(llm))
    if tokens is not None:
        assert model.auth is not None
        client._auth = AuthProvider(
            OAuthClientCredentials(token_url=TOKEN_URL, basic_key=AUTH_KEY, scope="API"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(tokens)),
        )
    return client


# --- wire bytes, hash and version ---------------------------------------------------


def test_chat_dialect_bytes_and_version_stay_as_before() -> None:
    model = _model()
    request = _request(model, tools=True)
    assert json.loads(request.body)["tools"][0]["type"] == "function"
    assert wire_version(model).startswith("chat-json-v1:")
    assert "+" not in wire_version(model).split(":")[0]


def test_functions_dialect_is_applied_before_hashing() -> None:
    chat = _request(_model(), tools=True)
    functions = _request(_model(dialect="functions"), tools=True)
    expected = to_functions_request(json.loads(chat.body))
    assert json.loads(functions.body) == expected
    assert (
        functions.body == json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode()
    )
    assert body_sha256(functions) != body_sha256(chat)
    assert functions.wire_version != chat.wire_version


def test_chat_dialect_drops_the_functions_state_id_from_history() -> None:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
        "functions_state_id": "state-1",
    }
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    chat = build_wire_request(
        _model(), messages, [], reasoning_effort=None, max_tokens=10, stream=False
    )
    assert "functions_state_id" not in chat.body.decode()
    functions = build_wire_request(
        _model(dialect="functions"),
        messages,
        [],
        reasoning_effort=None,
        max_tokens=10,
        stream=False,
    )
    assert json.loads(functions.body)["messages"][0]["functions_state_id"] == "state-1"


# --- headers ----------------------------------------------------------------------------


async def test_legacy_bearer_key_is_sent_as_before() -> None:
    seen: list[httpx.Request] = []

    def llm(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok())

    model = _model(api_key="legacy-key")
    await _client(model, llm).send(_request(model))
    assert seen[0].headers["authorization"] == "Bearer legacy-key"


async def test_header_key_and_extra_headers_are_sent() -> None:
    seen: list[httpx.Request] = []

    def llm(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok())

    auth = HeaderKeyAuthSpec(type="header_key", key_env="K", prefix="Api-Key ")
    model = _model(auth=auth, auth_secret="cloud-key", headers={"x-folder-id": "folder42"})
    await _client(model, llm).send(_request(model))
    assert seen[0].headers["authorization"] == "Api-Key cloud-key"
    assert seen[0].headers["x-folder-id"] == "folder42"
    assert seen[0].headers["content-type"] == "application/json"
    assert b"cloud-key" not in seen[0].content


@pytest.mark.parametrize(
    ("auth", "headers"),
    [
        (None, {"Content-Type": "text/plain"}),
        (BearerAuthSpec(type="bearer", key_env="K"), {"AUTHORIZATION": "Bearer other"}),
        (HeaderKeyAuthSpec(type="header_key", key_env="K", header="X-Api-Key"), {"x-api-key": "x"}),
        (None, {"Transfer-Encoding": "chunked"}),
    ],
)
async def test_extra_headers_cannot_override_auth_or_framing(
    auth: Any, headers: dict[str, str]
) -> None:
    model = _model(auth=auth, auth_secret="s", headers=headers)
    client = _client(model, lambda request: httpx.Response(200, json=_ok()))
    with pytest.raises(LLMError, match="must not set"):
        await client.send(_request(model))


# --- token refresh after 401 ------------------------------------------------------------


async def test_401_refreshes_the_token_once_and_both_attempts_are_audited() -> None:
    tokens = TokenServer()
    auths: list[str] = []

    def llm(request: httpx.Request) -> httpx.Response:
        auths.append(request.headers["authorization"])
        if len(auths) == 1:
            return httpx.Response(401, json={"message": "expired"})
        return httpx.Response(200, json=_ok())

    records: list[AttemptRecord] = []

    async def on_attempt(record: AttemptRecord) -> None:
        records.append(record)

    model = _oauth_model()
    result = await _client(model, llm, tokens).send(_request(model), on_attempt=on_attempt)
    assert result.content == "ok"
    assert auths == ["Bearer token-1", "Bearer token-2"]
    assert len(tokens.requests) == 2
    assert [(r.attempt, r.status, r.error_kind) for r in records] == [
        (0, "started", None),
        (0, "error", "auth"),
        (1, "started", None),
        (1, "completed", None),
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_second_401_is_an_error_without_a_third_request(stream: bool) -> None:
    tokens = TokenServer()
    calls: list[int] = []

    def llm(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"message": "no access"})

    model = _oauth_model()
    request = _request(model, stream=stream)

    async def on_delta(kind: str, text: str) -> None:
        return None

    with pytest.raises(LLMError) as caught:
        await _client(model, llm, tokens).send(request, on_delta=on_delta if stream else None)
    assert len(calls) == 2
    assert len(tokens.requests) == 2
    assert AUTH_KEY not in str(caught.value)
    assert "token-" not in str(caught.value)


def test_a_late_invalidate_keeps_the_token_of_a_parallel_refresh() -> None:
    provider = AuthProvider(
        OAuthClientCredentials(token_url=TOKEN_URL, basic_key=AUTH_KEY, scope="API")
    )
    from agent.provider_auth import _CachedToken

    provider._cached = _CachedToken("token-2", None)
    provider.invalidate("token-1")  # This request saw the old token.
    assert provider.token == "token-2"
    provider.invalidate("token-2")
    assert provider.token is None


async def test_provider_error_text_never_shows_the_token() -> None:
    tokens = TokenServer()

    def llm(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad token-1 here"}})

    model = _oauth_model()
    with pytest.raises(LLMError) as caught:
        await _client(model, llm, tokens).send(_request(model))
    assert "token-1" not in str(caught.value)


# --- functions dialect replies ------------------------------------------------------------


async def test_functions_reply_becomes_tool_calls_with_the_state_id() -> None:
    reply = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "function_call": {"name": "read_file", "arguments": {"path": "a.txt"}},
                    "functions_state_id": "state-7",
                },
                "finish_reason": "function_call",
            }
        ]
    }
    model = _model(dialect="functions")
    result = await _client(model, lambda request: httpx.Response(200, json=reply)).send(
        _request(model, tools=True)
    )
    assert result.finish_reason == "tool_calls"
    call = result.tool_calls[0]
    assert call["function"] == {"name": "read_file", "arguments": '{"path": "a.txt"}'}
    assert call["functions_state_id"] == "state-7"


async def test_functions_stream_becomes_tool_calls() -> None:
    chunk = {
        "choices": [
            {
                "delta": {"function_call": {"name": "read_file", "arguments": {"path": "a"}}},
                "finish_reason": "function_call",
            }
        ]
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
    model = _model(dialect="functions")

    async def on_delta(kind: str, text: str) -> None:
        return None

    result = await _client(
        model,
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    ).send(_request(model, stream=True, tools=True), on_delta=on_delta)
    assert result.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"path": "a"}


# --- certificate ---------------------------------------------------------------------------


def test_missing_ca_bundle_is_an_error_not_an_unverified_connection(tmp_path: Path) -> None:
    client = LLMClient(_model(ca_bundle=str(tmp_path / "missing.pem")))
    with pytest.raises(LLMError, match="CA bundle"):
        client._client()


def test_ca_bundle_gives_a_verifying_ssl_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ssl

    # The test must not depend on proxy settings of the machine that runs it.
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    import agent.llm as llm_module

    captured: dict[str, Any] = {}
    real = llm_module.httpx.AsyncClient

    def spy(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        captured.update(kwargs)
        return real(*args, **kwargs)

    bundle = tmp_path / "root.pem"
    bundle.write_text(
        Path(ssl.get_default_verify_paths().cafile or "").read_text()
        if ssl.get_default_verify_paths().cafile
        else "",
        encoding="utf-8",
    )
    if not bundle.read_text():
        pytest.skip("no system CA file to copy")
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", spy)
    LLMClient(_model(ca_bundle=str(bundle)))._client()
    context = captured["verify"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED


# --- review regressions: per-attempt secrets, SSE errors, stream state, repeated calls


def probe_model(*, oauth: bool = False, dialect: str = "chat") -> ResolvedModel:
    return ResolvedModel(
        id="probe",
        provider_id="probe",
        kind="openai",
        base_url="https://example.test/v1",
        api_key="",
        configured=True,
        model="probe",
        context_window=16000,
        max_tokens=1000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
        auth=OAuthAuthSpec(
            type="oauth_client_credentials",
            token_url="https://auth.test/token",
            key_env="PROBE_KEY",
            scope="API",
        )
        if oauth
        else None,
        auth_secret="synthetic-basic-key" if oauth else "",
        dialect=dialect,
    )


def probe_request(selected: ResolvedModel, *, stream: bool = False) -> Any:
    return build_wire_request(
        selected,
        [{"role": "user", "content": "probe"}],
        [],
        reasoning_effort=None,
        max_tokens=100,
        stream=stream,
    )


async def no_delta(kind: str, text: str) -> None:
    pass


@pytest.mark.parametrize("stream", [False, True])
async def test_error_redacts_the_token_used_by_the_attempt_after_parallel_refresh(
    stream: bool,
) -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    issued = 0
    sent = 0
    old_token = "synthetic-access-token-1"

    def token_handler(incoming: httpx.Request) -> httpx.Response:
        nonlocal issued
        issued += 1
        return httpx.Response(
            200,
            json={"access_token": f"synthetic-access-token-{issued}", "expires_at": 4102444800000},
        )

    async def llm_handler(incoming: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        if sent == 1:
            assert incoming.headers["authorization"] == f"Bearer {old_token}"
            first_entered.set()
            await release_first.wait()
            return httpx.Response(400, json={"error": {"message": f"rejected {old_token}"}})
        if sent == 2:
            return httpx.Response(401, json={"error": "expired"})
        assert incoming.headers["authorization"] == "Bearer synthetic-access-token-2"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    selected = probe_model(oauth=True)
    client = LLMClient(selected)
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(token_handler)) as auth_http,
        httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_http,
    ):
        client._http = llm_http
        client._auth = AuthProvider(
            OAuthClientCredentials(
                token_url="https://auth.test/token",
                basic_key="synthetic-basic-key",
                scope="API",
            ),
            client=auth_http,
        )
        first = asyncio.create_task(
            client.send(
                probe_request(selected, stream=stream), on_delta=no_delta if stream else None
            )
        )
        try:
            await asyncio.wait_for(first_entered.wait(), 2)
            await client.send(probe_request(selected))
            assert issued == 2
        finally:
            release_first.set()
        with pytest.raises(LLMError) as caught:
            await first
        assert old_token not in str(caught.value)


@pytest.mark.parametrize("late_state", [False, True])
async def test_functions_stream_retains_state_id(late_state: bool) -> None:
    delta: dict[str, Any] = {"function_call": {"name": "read_file", "arguments": {"path": "a"}}}
    if not late_state:
        delta["functions_state_id"] = "synthetic-state-7"
    chunks = [{"choices": [{"delta": delta, "finish_reason": None}]}]
    chunks.append(
        {
            "choices": [
                {
                    "delta": {"functions_state_id": "synthetic-state-7"} if late_state else {},
                    "finish_reason": "function_call",
                }
            ]
        }
    )
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    selected = probe_model(dialect="functions")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda incoming: httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        )
    ) as llm_http:
        client = LLMClient(selected)
        client._http = llm_http
        result = await client.send(probe_request(selected, stream=True), on_delta=no_delta)
    assert result.tool_calls[0].get("functions_state_id") == "synthetic-state-7"


async def test_sse_error_redacts_the_active_token() -> None:
    selected = probe_model(oauth=True)
    token = "synthetic-sse-access-token"
    body = f"data: {json.dumps({'error': {'message': f'rejected {token}'}})}\n\n"
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda incoming: httpx.Response(
                    200, json={"access_token": token, "expires_at": 4102444800000}
                )
            )
        ) as auth_http,
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda incoming: httpx.Response(
                    200, content=body, headers={"content-type": "text/event-stream"}
                )
            )
        ) as llm_http,
    ):
        client = LLMClient(selected)
        client._http = llm_http
        client._auth = AuthProvider(
            OAuthClientCredentials(
                token_url="https://auth.test/token",
                basic_key="synthetic-basic-key",
                scope="API",
            ),
            client=auth_http,
        )
        with pytest.raises(LLMError) as caught:
            await client.send(probe_request(selected, stream=True), on_delta=no_delta)
    assert token not in str(caught.value)


async def test_repeated_function_call_keeps_its_own_result_in_history() -> None:
    selected = probe_model(dialect="functions")
    count = 0

    def handler(incoming: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "functions_state_id": f"state-{count}",
                            "function_call": {"name": "read_file", "arguments": {"path": "a"}},
                        },
                        "finish_reason": "function_call",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as llm_http:
        client = LLMClient(selected)
        client._http = llm_http
        first = await client.send(probe_request(selected))
        second = await client.send(probe_request(selected))
    messages = [
        {"role": "assistant", "content": None, "tool_calls": first.tool_calls},
        {"role": "tool", "tool_call_id": first.tool_calls[0]["id"], "content": "old result"},
        {"role": "assistant", "content": None, "tool_calls": second.tool_calls},
        {"role": "tool", "tool_call_id": second.tool_calls[0]["id"], "content": "new result"},
    ]
    wire = build_wire_request(
        selected, messages, [], reasoning_effort=None, max_tokens=100, stream=False
    )
    results = [item["content"] for item in wire.payload["messages"] if item["role"] == "function"]
    assert results == ["old result", "new result"]


# --- streamed arguments in fragments


@pytest.mark.parametrize("empty_tail", [False, True])
async def test_name_only_delta_does_not_insert_arguments(empty_tail: bool) -> None:
    selected = ResolvedModel(
        id="probe",
        provider_id="probe",
        kind="openai",
        base_url="https://example.test/v1",
        api_key="",
        configured=True,
        model="probe",
        context_window=16000,
        max_tokens=1000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
        dialect="functions",
    )
    deltas = [
        {"function_call": {"name": "read_file"}},
        {"function_call": {"arguments": '{"path":'}},
        {"function_call": {"arguments": '"a.txt"}'}},
    ]
    if empty_tail:
        deltas.append({"function_call": {}})
    deltas.append({"functions_state_id": "state-7"})
    chunks = [{"choices": [{"delta": delta, "finish_reason": None}]} for delta in deltas]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "function_call"}]})
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    body += "data: [DONE]\n\n"
    client = LLMClient(selected)
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )
    )
    request = build_wire_request(
        selected,
        [{"role": "user", "content": "probe"}],
        [],
        reasoning_effort=None,
        max_tokens=100,
        stream=True,
    )

    async def on_delta(kind: str, text: str) -> None:
        pass

    try:
        reply = await client.send(request, on_delta=on_delta)
    finally:
        await client.aclose()
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call["function"]["name"] == "read_file"
    assert call["functions_state_id"] == "state-7"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


async def test_streamed_whole_argument_object_still_works() -> None:
    selected = probe_model(dialect="functions")
    chunks = [
        {
            "choices": [
                {"delta": {"function_call": {"name": "read_file", "arguments": {"path": "b"}}}}
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "function_call"}]},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    client = LLMClient(selected)
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda incoming: httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )
    )
    reply = await client.send(probe_request(selected, stream=True), on_delta=no_delta)
    await client.aclose()
    assert json.loads(reply.tool_calls[0]["function"]["arguments"]) == {"path": "b"}
