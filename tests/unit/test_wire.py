from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from agent.model_config import ReasoningSpec, ResolvedModel
from agent.wire import (
    ESTIMATOR_VERSION,
    WireRequest,
    body_sha256,
    build_wire_request,
    endpoint_of,
    estimate_wire,
    wire_version,
)


def make_model(
    *,
    base_url: str = "https://llm.example.com/v1",
    wire: Literal["reasoning_effort", "thinking_budget", "enable_thinking"] | None = None,
    efforts: list[str] | None = None,
    default: str | None = None,
) -> ResolvedModel:
    allowed = efforts or []
    return ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url=base_url,
        api_key="",
        configured=True,
        model="m",
        context_window=32_000,
        max_tokens=4_000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(
            supported=bool(allowed),
            allowed_efforts=allowed,
            default_effort=default,
            wire_parameter=wire,
        ),
        pricing=None,
    )


def messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
    ]


def tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
        }
    ]


def test_endpoint_of() -> None:
    m = make_model(base_url="https://llm.example.com/v1/")
    assert endpoint_of(m) == "https://llm.example.com/v1/chat/completions"


def test_payload_without_tools() -> None:
    m = make_model()
    request = build_wire_request(
        m, messages(), [], reasoning_effort=None, max_tokens=100, stream=False
    )
    assert request.payload == {"model": "m", "messages": messages(), "max_tokens": 100}


def test_payload_with_tools() -> None:
    m = make_model()
    request = build_wire_request(
        m, messages(), tools(), reasoning_effort=None, max_tokens=100, stream=False
    )
    assert request.payload["tools"] == tools()
    assert request.payload["tool_choice"] == "auto"


def test_reasoning_reasoning_effort() -> None:
    m = make_model(wire="reasoning_effort", efforts=["low", "high"], default="high")
    request = build_wire_request(
        m, messages(), [], reasoning_effort="low", max_tokens=100, stream=False
    )
    assert request.payload["reasoning_effort"] == "low"


def test_reasoning_default_when_not_given() -> None:
    m = make_model(wire="reasoning_effort", efforts=["low", "high"], default="high")
    request = build_wire_request(
        m, messages(), [], reasoning_effort=None, max_tokens=100, stream=False
    )
    assert request.payload["reasoning_effort"] == "high"


def test_reasoning_thinking_dialect() -> None:
    m = make_model(wire="thinking_budget", efforts=["low", "medium", "high"], default="high")
    request = build_wire_request(
        m, messages(), [], reasoning_effort="medium", max_tokens=4_000, stream=False
    )
    assert request.payload["thinking"] == {"type": "enabled", "budget_tokens": 3_999}


def test_invalid_effort_adds_no_fields() -> None:
    m = make_model(wire="reasoning_effort", efforts=["low", "high"], default="low")
    request = build_wire_request(
        m, messages(), [], reasoning_effort="banana", max_tokens=100, stream=False
    )
    assert "reasoning_effort" not in request.payload
    assert "thinking" not in request.payload


def test_stream_fields() -> None:
    m = make_model()
    request = build_wire_request(
        m, messages(), [], reasoning_effort=None, max_tokens=100, stream=True
    )
    assert request.payload["stream"] is True
    assert request.payload["stream_options"] == {"include_usage": True}


def test_body_matches_payload() -> None:
    m = make_model(wire="reasoning_effort", efforts=["low"], default="low")
    request = build_wire_request(
        m, messages(), tools(), reasoning_effort="low", max_tokens=100, stream=True
    )
    expected = json.dumps(request.payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    assert request.body == expected


def test_inputs_do_not_mutate_request() -> None:
    m = make_model(wire="reasoning_effort", efforts=["low"], default="low")
    msgs = messages()
    tls = tools()
    request = build_wire_request(m, msgs, tls, reasoning_effort="low", max_tokens=100, stream=True)
    msgs[0]["content"] = "changed"
    tls.clear()
    assert request.payload["messages"][0]["content"] == "s"
    assert request.payload["tools"] == tools()


def test_estimate_independent_of_model_max_tokens_stream() -> None:
    msg = messages()
    plain = make_model()
    other = make_model(base_url="https://other.example/v1")
    a = build_wire_request(plain, msg, [], reasoning_effort=None, max_tokens=100, stream=False)
    b = build_wire_request(other, msg, [], reasoning_effort=None, max_tokens=9_999, stream=True)
    assert estimate_wire(a) == estimate_wire(b)


def test_estimate_grows_with_tools_and_reasoning() -> None:
    base = build_wire_request(
        make_model(), messages(), [], reasoning_effort=None, max_tokens=100, stream=False
    )
    with_tools = build_wire_request(
        make_model(), messages(), tools(), reasoning_effort=None, max_tokens=100, stream=False
    )
    reasoned = make_model(wire="reasoning_effort", efforts=["low"], default="low")
    with_reasoning = build_wire_request(
        reasoned, messages(), [], reasoning_effort="low", max_tokens=100, stream=False
    )
    assert estimate_wire(with_tools) > estimate_wire(base)
    assert estimate_wire(with_reasoning) > estimate_wire(base)


def test_estimate_exact_formula() -> None:
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    request = WireRequest(payload=payload, body=b"", wire_version="")
    assert estimate_wire(request) == 24


def test_wire_version_scope() -> None:
    a = make_model(base_url="https://a.example/v1")
    b = make_model(base_url="https://b.example/v1")
    assert wire_version(a) != wire_version(b)
    assert wire_version(make_model(base_url="https://x/v1/")) == wire_version(
        make_model(base_url="https://x/v1")
    )
    assert wire_version(a).startswith(ESTIMATOR_VERSION + ":")
    assert len(wire_version(a).split(":", 1)[1]) == 64


def test_body_sha256() -> None:
    m = make_model()
    request = build_wire_request(
        m, messages(), [], reasoning_effort=None, max_tokens=100, stream=False
    )
    assert body_sha256(request) == hashlib.sha256(request.body).hexdigest()
