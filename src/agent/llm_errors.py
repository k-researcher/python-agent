"""Typed classification of LLM provider errors, including context overflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

import httpx

from agent.redaction import redact


class LLMErrorKind(StrEnum):
    """Typed categories for provider errors."""

    context_overflow = "context_overflow"
    rate_limit = "rate_limit"
    auth = "auth"
    not_found = "not_found"
    bad_request = "bad_request"
    server = "server"
    timeout = "timeout"
    network = "network"
    unknown = "unknown"


@dataclass(frozen=True, slots=True)
class ProviderErrorInfo:
    """A normalized, safe description of one provider error."""

    kind: LLMErrorKind
    status: int | None
    code: str | None
    message: str
    retryable: bool


_CONTEXT_CODES = frozenset(
    {
        "context_length_exceeded",
        "context_window_exceeded",
        "contextwindowexceedederror",
        "string_above_max_length",
    }
)

_CONTEXT_PHRASES = (
    "maximum context length",
    "context length exceeded",
    "context window",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "exceeds the context",
    "contextwindowexceedederror",
    "reduce the length of the messages",
)


def _as_str(value: object) -> str | None:
    """Return the value as str, or None for null values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def parse_error_body(body: str) -> tuple[str | None, str | None, str | None]:
    """Return (code, type, message) from an OpenAI/LiteLLM style JSON error body."""
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None

    error = data.get("error")
    if isinstance(error, dict):
        return (
            _as_str(error.get("code")),
            _as_str(error.get("type")),
            _as_str(error.get("message")),
        )
    if isinstance(error, str):
        return None, None, error
    for key in ("detail", "message"):
        value = data.get(key)
        if isinstance(value, str):
            return None, None, value
    return None, None, None


def _is_context_marker(value: str | None) -> bool:
    """Return True for a code or type from the known context overflow set."""
    return value is not None and value.lower() in _CONTEXT_CODES


def _has_context_phrase(message: str) -> bool:
    """Return True when the message contains a known context overflow phrase."""
    lowered = message.lower()
    return any(phrase in lowered for phrase in _CONTEXT_PHRASES)


def _info(
    kind: LLMErrorKind,
    status: int | None,
    code: str | None,
    message: str,
    retryable: bool,
) -> ProviderErrorInfo:
    return ProviderErrorInfo(kind, status, code, message, retryable)


def classify_http_error(status: int, body: str) -> ProviderErrorInfo:
    """Classify an HTTP error response into a typed, safe provider error."""
    code, type_, message = parse_error_body(body)
    message = redact(message or body)[:500]
    if _is_context_marker(code) or _is_context_marker(type_):
        return _info(LLMErrorKind.context_overflow, status, code, message, False)
    if status in (400, 413, 422) and _has_context_phrase(message):
        return _info(LLMErrorKind.context_overflow, status, code, message, False)
    if status in (401, 403):
        return _info(LLMErrorKind.auth, status, code, message, False)
    if status == 404:
        return _info(LLMErrorKind.not_found, status, code, message, False)
    if status == 408:
        return _info(LLMErrorKind.timeout, status, code, message, True)
    if status == 429:
        return _info(LLMErrorKind.rate_limit, status, code, message, True)
    if 500 <= status <= 599:
        return _info(LLMErrorKind.server, status, code, message, True)
    if 400 <= status <= 499:
        return _info(LLMErrorKind.bad_request, status, code, message, False)
    return _info(LLMErrorKind.unknown, status, code, message, False)


def classify_exception(exc: BaseException) -> ProviderErrorInfo:
    """Classify a transport exception (timeout or network) into a typed error."""
    if isinstance(exc, httpx.TimeoutException):
        kind, retryable = LLMErrorKind.timeout, True
    elif isinstance(exc, httpx.TransportError):
        kind, retryable = LLMErrorKind.network, True
    else:
        kind, retryable = LLMErrorKind.unknown, False
    message = redact(f"{type(exc).__name__}: {exc}")[:500]
    return _info(kind, None, None, message, retryable)


def is_context_overflow(info: ProviderErrorInfo) -> bool:
    """Return True when the error describes a context overflow.

    Only the classified kind counts: a known phrase in another error is not an overflow.
    """
    return info.kind is LLMErrorKind.context_overflow
