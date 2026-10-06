"""Build a deterministic wire request for one chat-completions call and estimate its token size."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from agent.model_config import ResolvedModel
from agent.reasoning import ReasoningDecision, wire_parameters

ESTIMATOR_VERSION = "chat-json-v1"
_EXCLUDED_FIELDS = frozenset({"model", "max_tokens", "stream", "stream_options"})


@dataclass(frozen=True, slots=True)
class WireRequest:
    payload: dict[str, Any]  # do not change after build
    body: bytes  # final JSON bytes; send and hash exactly these
    wire_version: str


def endpoint_of(model: ResolvedModel) -> str:
    """Return base_url without the trailing slash plus "/chat/completions"."""
    return f"{model.base_url.rstrip('/')}/chat/completions"


def wire_version(model: ResolvedModel) -> str:
    """Return ESTIMATOR_VERSION plus the sha256 hex of the normalized endpoint."""
    normalized = endpoint_of(model).lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{ESTIMATOR_VERSION}:{digest}"


def build_wire_request(
    model: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    reasoning_effort: str | None,
    max_tokens: int,
    stream: bool,
) -> WireRequest:
    """Build the payload, its JSON body and the wire version for one call."""
    payload: dict[str, Any] = {
        "model": model.model,
        "messages": copy.deepcopy(messages),
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = copy.deepcopy(tools)
        payload["tool_choice"] = "auto"
    effort = reasoning_effort or model.reasoning.default_effort
    if effort in model.reasoning.allowed_efforts:
        decision = ReasoningDecision(effort, effort, "message", "caller")
        payload.update(wire_parameters(model, decision, max_tokens))
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return WireRequest(payload=payload, body=body, wire_version=wire_version(model))


def estimate_wire(request: WireRequest) -> int:
    """Estimate request token count from the non-excluded payload fields."""
    projection = {
        key: value for key, value in request.payload.items() if key not in _EXCLUDED_FIELDS
    }
    json_chars = len(
        json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    messages = projection.get("messages", [])
    tools = projection.get("tools", [])
    return math.ceil(json_chars / 4) + 8 + 4 * len(messages) + 8 * len(tools)


def body_sha256(request: WireRequest) -> str:
    """Return the hex sha256 digest of the exact request body."""
    return hashlib.sha256(request.body).hexdigest()
