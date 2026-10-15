"""Integration regressions for ContextInference.

These tests exercise the real LLMClient.send() path over httpx.MockTransport and the
real database audit/calibration persistence. No network request leaves the process and
no live environment data is read; each test builds its own isolated project/session
inside the conftest database.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from agent import context_inference as ctxmod
from agent.context_inference import (
    MAX_HTTP_ATTEMPTS,
    ContextInference,
)
from agent.context_manager import ContextBudgetError
from agent.database import init_database, session_factory
from agent.llm import LLMClient, LLMError
from agent.model_config import ReasoningSpec, ResolvedModel, ResolvedModelRegistry
from agent.models import Event, Message, OutboundAudit, Project, Session, TokenCalibration
from agent.reasoning import ReasoningDecision


@pytest.fixture(autouse=True)
def _guard_llm_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject any LLM HTTP attempt that lacks an explicit MockTransport."""

    def _no_escape(self: LLMClient) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            raise AssertionError("Network escape: LLM client without MockTransport")
        if not isinstance(self._http._transport, httpx.MockTransport):
            raise AssertionError("Network escape: real HTTP transport in LLM test")
        return self._http

    monkeypatch.setattr(ctxmod.LLMClient, "_client", _no_escape)


def _model(
    model_id: str,
    *,
    window: int = 8_000,
    max_tokens: int = 1_000,
    configured: bool = True,
    supported: bool = False,
    max_retries: int = 0,
    provider_id: str = "openai",
) -> ResolvedModel:
    return ResolvedModel(
        id=model_id,
        provider_id=provider_id,
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test",
        configured=configured,
        model=f"model-{model_id}",
        context_window=window,
        max_tokens=max_tokens,
        timeout_seconds=30,
        max_retries=max_retries,
        reasoning=ReasoningSpec(supported=supported),
        pricing=None,
    )


def _registry(
    primary: ResolvedModel,
    fallback: tuple[ResolvedModel, ...] = (),
    *,
    light: ResolvedModel | None = None,
) -> ResolvedModelRegistry:
    models = {primary.id: primary}
    for model in fallback:
        models[model.id] = model
    if light is not None:
        models[light.id] = light
    return ResolvedModelRegistry(
        models=models,
        default_id=primary.id,
        fallback=[model.id for model in fallback],
        roles={},
        warnings=[],
        source="yaml",
        checksum="test",
        light_id=light.id if light is not None else None,
    )


def _decision() -> ReasoningDecision:
    return ReasoningDecision(None, None, "none", "model")


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response], model: ResolvedModel
) -> LLMClient:
    client = LLMClient(model)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10)
    return client


def _make_client_factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable:
    def factory(model: ResolvedModel) -> LLMClient:
        return _make_client(handler, model)

    return factory


def _stream_response(usage: dict[str, Any] | None = None) -> bytes:
    body: dict[str, Any] = {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in [body])


def _json_error(status: int, message: str, code: str | None = None) -> httpx.Response:
    error: dict[str, Any] = {"message": message}
    if code is not None:
        error["code"] = code
    return httpx.Response(status, json={"error": error})


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


async def _create_session(tmp_path: Path, name: str) -> str:
    """Build a running project/session in the shared database; return the session id."""
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name=name, root_path=str(tmp_path / name))
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            title=name,
            configuration={},
            status="running",
        )
        db.add(session)
        await db.commit()
        return session.id


async def _add_messages(session_id: str, rows: list[dict[str, Any]]) -> None:
    async with session_factory() as db:
        messages = [Message(session_id=session_id, **row) for row in rows]
        db.add_all(messages)
        await db.commit()


async def _audit_rows(session_id: str) -> list[OutboundAudit]:
    async with session_factory() as db:
        result = await db.scalars(
            select(OutboundAudit)
            .where(OutboundAudit.session_id == session_id)
            .order_by(OutboundAudit.id)
        )
        return list(result)


async def _event_types(session_id: str) -> list[str]:
    async with session_factory() as db:
        result = await db.scalars(
            select(Event).where(Event.session_id == session_id).order_by(Event.id)
        )
        return [event.type for event in result]


async def _search_event(session_id: str, event_type: str) -> list[Event]:
    async with session_factory() as db:
        result = await db.scalars(
            select(Event)
            .where(Event.session_id == session_id, Event.type == event_type)
            .order_by(Event.id)
        )
        return list(result)


async def _calibration_rows(provider: str) -> list[TokenCalibration]:
    async with session_factory() as db:
        result = await db.scalars(
            select(TokenCalibration).where(TokenCalibration.provider == provider)
        )
        return list(result)


def _simple_history() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"},
    ]


def _compressible_history(count: int = 12, chunk: int = 1000) -> list[dict[str, Any]]:
    """A long history whose middle exchanges are compressible.

    The leading system row, the framing user messages, and the two most recent complete
    exchanges are protected; the exchanges between them form a contiguous compressible
    prefix that summary stages can replace with a small checkpoint.
    """
    rows: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "start the task"},
    ]
    for index in range(count):
        rows.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {"path": f"/tmp/f{index}", "content": "x" * chunk}
                            ),
                        },
                    }
                ],
            }
        )
        rows.append(
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "content": json.dumps({"ok": True, "data": "r" * chunk}),
            }
        )
    rows.append({"role": "user", "content": "wrap it up"})
    return rows


def _valid_summary() -> str:
    return (
        "## Goal\nfinish\n"
        "## Constraints\n- (none)\n"
        "## Decisions\n- (none)\n"
        "## Changed paths\n- /tmp/x\n"
        "## Check results\n- (none)\n"
        "## Open tasks\n- (none)"
    )


def _inference(
    session_id: str,
    models: ResolvedModelRegistry,
    client_factory: Callable,
    *,
    primary: ResolvedModel,
) -> ContextInference:
    return ContextInference(
        session_id,
        0,
        [],
        models,
        client_factory,
        lambda _model: _decision(),
    )


async def test_audit_records_exact_body_and_stores_present_usage(tmp_path: Path) -> None:
    sent: list[tuple[bytes, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((request.content, _sha256(request.content)))
        return httpx.Response(
            200,
            content=_stream_response(usage={"prompt_tokens": 7, "completion_tokens": 3}),
        )

    primary = _model("primary", provider_id="prov-audit-usage")
    session_id = await _create_session(tmp_path, "audit-usage")
    await _add_messages(session_id, _simple_history())
    inference = _inference(
        session_id,
        _registry(primary),
        _make_client_factory(handler),
        primary=primary,
    )

    result = await inference.run(primary)

    assert result.response.content == "hi"
    assert len(sent) == 1
    body, digest = sent[0]
    assert digest == _sha256(body)

    audits = await _audit_rows(session_id)
    assert len(audits) == 1
    audit = audits[0]
    assert audit.status == "completed"
    assert audit.purpose == "inference"
    assert audit.payload_bytes == len(body)
    assert audit.payload_sha256 == digest
    assert audit.prompt_tokens == 7
    assert audit.completion_tokens == 3
    assert audit.destination == "https://llm.example/v1/chat/completions"
    assert str(audit.inference_id) == inference.state.inference_id

    cals = await _calibration_rows("prov-audit-usage")
    assert len(cals) == 1
    assert cals[0].samples == 1
    assert cals[0].factor == pytest.approx((1 - 0.2) * 1.0 + 0.2 * 7 / audit.raw_estimate)


async def test_audit_nullable_usage_is_not_zeroed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_stream_response())

    primary = _model("primary", provider_id="prov-audit-null")
    session_id = await _create_session(tmp_path, "audit-null-usage")
    await _add_messages(session_id, _simple_history())
    inference = _inference(
        session_id,
        _registry(primary),
        _make_client_factory(handler),
        primary=primary,
    )

    await inference.run(primary)

    audits = await _audit_rows(session_id)
    assert len(audits) == 1
    assert audits[0].status == "completed"
    assert audits[0].prompt_tokens is None
    assert audits[0].completion_tokens is None

    cals = await _calibration_rows("prov-audit-null")
    assert cals == []


async def test_overflow_retries_produce_strictly_smaller_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []
    hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": _valid_summary()}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
            )
        nonlocal hits
        sizes.append(len(request.content))
        hits += 1
        if hits <= 2:
            return _json_error(
                400, "maximum context length exceeded", code="context_length_exceeded"
            )
        return httpx.Response(
            200,
            content=_stream_response(usage={"prompt_tokens": 3, "completion_tokens": 2}),
        )

    primary = _model("primary")
    session_id = await _create_session(tmp_path, "overflow-smaller")
    await _add_messages(session_id, _compressible_history())
    inference = _inference(
        session_id,
        _registry(primary),
        _make_client_factory(handler),
        primary=primary,
    )
    _patch_summary_llm(monkeypatch, handler)

    result = await inference.run(primary)

    assert result.response.content == "hi"
    assert len(sizes) == 3
    assert sizes[0] > sizes[1] > sizes[2]
    overflows = await _search_event(session_id, "context.overflow")
    assert len(overflows) == 2
    assert [event.payload["overflow_retry"] for event in overflows] == [1, 2]


async def test_overflow_protected_minimum_raises_budget_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": _valid_summary()}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
            )
        sizes.append(len(request.content))
        return _json_error(400, "context length exceeded", code="context_length_exceeded")

    primary = _model(
        "primary", window=4_000, max_tokens=1_000, provider_id="prov-overflow-protected"
    )
    light = _model(
        "light",
        window=16_000,
        max_tokens=2_000,
        supported=True,
        provider_id="prov-overflow-protected",
    )
    session_id = await _create_session(tmp_path, "overflow-protected")
    await _add_messages(session_id, _compressible_history(count=20, chunk=1000))
    inference = _inference(
        session_id,
        _registry(primary, light=light),
        _make_client_factory(handler),
        primary=primary,
    )
    _patch_summary_llm(monkeypatch, handler)

    with pytest.raises(ContextBudgetError) as caught:
        await inference.run(primary)
    error = caught.value
    assert isinstance(error, ContextBudgetError)
    assert "protected_over_budget" in error.reason

    assert len(sizes) == 1
    assert error.result.fits is False


async def test_fallback_only_on_transient_and_reprepares(tmp_path: Path) -> None:
    plan = {"primary": 1}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("model") == "model-primary" and plan["primary"] == 1:
            plan["primary"] -= 1
            return httpx.Response(503, text="overloaded")
        return httpx.Response(
            200,
            content=_stream_response(usage={"prompt_tokens": 2, "completion_tokens": 1}),
        )

    primary = _model("primary")
    fallback = _model("fallback")
    session_id = await _create_session(tmp_path, "fallback-transient")
    await _add_messages(session_id, _simple_history())
    inference = _inference(
        session_id,
        _registry(primary, (fallback,)),
        _make_client_factory(handler),
        primary=primary,
    )

    result = await inference.run(primary)

    assert result.model.id == "fallback"
    assert result.response.content == "hi"
    assert result.prepared.context.request.payload["model"] == "model-fallback"
    events = await _event_types(session_id)
    assert "llm.fallback" in events


async def test_non_transient_errors_never_fallback(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_error(401, "bad key")

    primary = _model("primary")
    fallback = _model("fallback")
    session_id = await _create_session(tmp_path, "fallback-auth")
    await _add_messages(session_id, _simple_history())
    inference = _inference(
        session_id,
        _registry(primary, (fallback,)),
        _make_client_factory(handler),
        primary=primary,
    )

    with pytest.raises(LLMError) as caught:
        await inference.run(primary)
    assert caught.value.kind == "auth"

    audits = await _audit_rows(session_id)
    assert len(audits) == 1
    events = await _event_types(session_id)
    assert "llm.fallback" not in events


async def test_content_delta_blocks_retry_output_increase_and_fallback(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1

        payload = _sse_inline_error()
        return httpx.Response(200, content=payload)

    primary = _model("primary")
    fallback = _model("fallback")
    session_id = await _create_session(tmp_path, "delta-blocked")
    await _add_messages(session_id, _simple_history())
    inference = _inference(
        session_id,
        _registry(primary, (fallback,)),
        _make_client_factory(handler),
        primary=primary,
    )

    with pytest.raises(LLMError) as caught:
        await inference.run(primary)
    assert caught.value.started is True

    assert calls == 1
    audits = await _audit_rows(session_id)
    assert len(audits) == 1
    assert audits[0].status == "error"
    events = await _event_types(session_id)
    assert "llm.fallback" not in events


async def test_max_http_attempts_caps_entire_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = 0

    async def _no_sleep(_seconds: float = 0.0) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(503, text="busy")

    primary = _model("primary", window=64_000, max_tokens=4_000, max_retries=20)
    fallback = _model("fallback", window=64_000, max_tokens=4_000, max_retries=20)
    session_id = await _create_session(tmp_path, "attempt-cap")
    await _add_messages(session_id, _simple_history())

    inference = _inference(
        session_id,
        _registry(primary, (fallback,)),
        _make_client_factory(handler),
        primary=primary,
    )

    with pytest.raises(LLMError):
        await inference.run(primary)

    audits = await _audit_rows(session_id)
    assert len(audits) == MAX_HTTP_ATTEMPTS
    assert hits == MAX_HTTP_ATTEMPTS
    assert all(audit.status == "error" for audit in audits)


async def test_preview_prepare_paid_summary_false_has_no_side_effects(
    tmp_path: Path,
) -> None:
    hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, content=_stream_response())

    primary = _model("primary", window=4_000, max_tokens=1_000, provider_id="prov-preview")
    light = _model("light", window=16_000, max_tokens=2_000, supported=True)
    session_id = await _create_session(tmp_path, "preview")
    await _add_messages(session_id, _compressible_history())
    models = _registry(primary, light=light)

    inference = _inference(
        session_id,
        models,
        _make_client_factory(handler),
        primary=primary,
    )

    prepared = await inference.prepare(primary, _decision(), paid_summary=False)

    assert prepared.context.stage != "summary"
    assert hits == 0
    assert await _audit_rows(session_id) == []
    assert await _calibration_rows("prov-preview") == []


def _sse_inline_error() -> bytes:
    chunks = [
        {"choices": [{"delta": {"content": "partial"}}]},
        {"error": {"message": "boom"}},
    ]
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)


def _patch_summary_llm(monkeypatch: pytest.MonkeyPatch, handler: Callable) -> None:
    """Make the dedicated summary LLMClient use the same mock transport."""
    monkeypatch.setattr(
        ctxmod,
        "LLMClient",
        lambda model: _make_client(handler, model),
    )


def _summary_handler(
    *,
    content: str | None = None,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> Callable:
    seen: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        purpose = "inference" if body.get("stream") else "summary"
        seen[purpose] = seen.get(purpose, 0) + 1
        if purpose == "inference":
            return httpx.Response(
                200,
                content=_stream_response(usage={"prompt_tokens": 2, "completion_tokens": 1}),
            )
        message: dict[str, Any] = {"content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        payload = {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        }
        return httpx.Response(200, json=payload)

    handler.seen = seen
    return handler


async def test_checkpoint_saved_before_primary_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = _model("primary", window=4_000, max_tokens=1_000, provider_id="prov-checkpoint-first")
    light = _model(
        "light",
        window=16_000,
        max_tokens=2_000,
        supported=True,
        provider_id="prov-checkpoint-first",
    )
    session_id = await _create_session(tmp_path, "checkpoint-first")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": _valid_summary()}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
            )
        return httpx.Response(
            200,
            content=_stream_response(usage={"prompt_tokens": 2, "completion_tokens": 1}),
        )

    models = _registry(primary, light=light)
    await _add_messages(session_id, _compressible_history(count=20, chunk=1000))
    inference = _inference(
        session_id,
        models,
        _make_client_factory(handler),
        primary=primary,
    )
    _patch_summary_llm(monkeypatch, handler)

    result = await inference.run(primary)

    async with session_factory() as db:
        rows = await db.scalars(
            select(Message).where(Message.session_id == session_id, Message.kind == "summary")
        )
        checkpoints = list(rows)
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint.covers_until is not None
    assert checkpoint.covers_until > 0
    assert checkpoint.source_hash is not None
    assert len(checkpoint.source_hash) == 64
    assert checkpoint.summary_version == 1

    assert result.response.content == "hi"
    audits = await _audit_rows(session_id)
    purposes = [audit.purpose for audit in audits]
    assert purposes[0] == "summary"
    assert purposes[1] == "inference"


async def test_summary_invalid_response_sets_summary_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _create_session(tmp_path, "summary-invalid")

    tool_calls: list[dict[str, Any]] = [
        {
            "id": "tc-1",
            "type": "function",
            "function": {"name": "write_file", "arguments": "{}"},
        }
    ]
    cases = [
        ("blank", "", None, "stop"),
        ("tool_calls", None, tool_calls, "stop"),
        ("length", "valid summary content", None, "length"),
    ]
    for index, (label, content, tc, finish) in enumerate(cases):
        provider = f"prov-summary-invalid-{index}"
        primary = _model("primary", window=4_000, max_tokens=1_000, provider_id=provider)
        light = _model(
            "light",
            window=16_000,
            max_tokens=2_000,
            supported=True,
            provider_id=provider,
        )
        handler = _summary_handler(content=content, tool_calls=tc, finish_reason=finish)
        models = _registry(primary, light=light)
        await _add_messages(session_id, _compressible_history(count=20, chunk=1000))
        inference = _inference(
            session_id,
            models,
            _make_client_factory(handler),
            primary=primary,
        )
        _patch_summary_llm(monkeypatch, handler)

        result = await inference.run(primary)

        assert result.response.content == "hi"
        failed = await _search_event(session_id, "context.summary_failed")
        assert failed, f"expected context.summary_failed for {label}"
        assert failed[-1].payload["kind"] in {"invalid_response", "budget"}
        assert handler.seen.get("summary", 0) >= 1
        assert handler.seen.get("inference", 0) == 1

        async with session_factory() as db:
            await db.execute(
                __import__("sqlalchemy").delete(Message).where(Message.session_id == session_id)
            )
            await db.commit()


async def test_cancel_during_summary_leaves_no_started_and_no_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": _valid_summary()}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
            )
        return httpx.Response(
            200,
            content=_stream_response(usage={"prompt_tokens": 2, "completion_tokens": 1}),
        )

    primary = _model("primary", window=4_000, max_tokens=1_000)
    light = _model("light", window=16_000, max_tokens=2_000, supported=True)
    session_id = await _create_session(tmp_path, "cancel-summary")
    await _add_messages(session_id, _compressible_history(count=8, chunk=500))
    models = _registry(primary, light=light)
    inference = _inference(
        session_id,
        models,
        _make_client_factory(handler),
        primary=primary,
    )
    _patch_summary_llm(monkeypatch, handler)

    async with session_factory() as db:
        session = await db.get(Session, session_id)
        session.stop_requested = True
        await db.commit()

    with pytest.raises(asyncio.CancelledError):
        await inference.run(primary)

    audits = await _audit_rows(session_id)

    assert audits == []
    async with session_factory() as db:
        rows = await db.scalars(
            select(Message).where(Message.session_id == session_id, Message.kind == "summary")
        )
        assert list(rows) == []
