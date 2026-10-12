from __future__ import annotations

import json

import httpx
import pytest

from agent.llm_errors import (
    LLMErrorKind,
    ProviderErrorInfo,
    classify_exception,
    classify_http_error,
    is_context_overflow,
    parse_error_body,
)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"error": {"message": "boom", "type": "t", "code": "c"}}', ("c", "t", "boom")),
        ('{"error": "plain text"}', (None, None, "plain text")),
        ('{"detail": "detail text"}', (None, None, "detail text")),
        ('{"message": "message text"}', (None, None, "message text")),
    ],
)
def test_parse_error_formats(
    body: str, expected: tuple[str | None, str | None, str | None]
) -> None:
    assert parse_error_body(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        "",
        "not json",
        "[]",
        "null",
        '"just a string"',
    ],
)
def test_parse_error_unparseable(body: str) -> None:
    assert parse_error_body(body) == (None, None, None)


@pytest.mark.parametrize(
    ("status", "phrase"),
    [
        (400, "this model's maximum context length is 128000 tokens"),
        (400, "the message goes beyond context length exceeded"),
        (413, "request is too large for the context window"),
        (422, "too many tokens in your prompt"),
        (400, "prompt is too long"),
        (413, "input is too long"),
        (400, "this exceeds the context limits"),
        (422, "litellm.ContextWindowExceededError: reduce the length of the messages"),
        (400, "please reduce the length of the messages"),
        (413, "contextwindowexceedederror here"),
    ],
)
def test_context_phrase(status: int, phrase: str) -> None:
    info = classify_http_error(status, phrase)
    assert info.kind is LLMErrorKind.context_overflow
    assert info.status == status
    assert not info.retryable
    assert is_context_overflow(info)


@pytest.mark.parametrize(
    "code",
    [
        "context_length_exceeded",
        "context_window_exceeded",
        "ContextWindowExceededError",
        "string_above_max_length",
    ],
)
def test_context_code(code: str) -> None:
    body = json.dumps({"error": {"message": "nope", "type": "invalid_request_error", "code": code}})
    info = classify_http_error(400, body)
    assert info.kind is LLMErrorKind.context_overflow
    assert info.code == code
    assert not info.retryable


def test_context_type() -> None:
    body = '{"error": {"message": "nope", "type": "context_length_exceeded", "code": null}}'
    info = classify_http_error(400, body)
    assert info.kind is LLMErrorKind.context_overflow


@pytest.mark.parametrize("status", [400, 500])
def test_context_code_any_status(status: int) -> None:
    info = classify_http_error(status, '{"error": {"code": "context_length_exceeded"}}')
    assert info.kind is LLMErrorKind.context_overflow
    assert info.status == status


def test_litellm_text() -> None:
    info = classify_http_error(400, "litellm.ContextWindowExceededError: context exceeded")
    assert info.kind is LLMErrorKind.context_overflow


@pytest.mark.parametrize(
    "status",
    [400, 404, 422],
)
def test_not_context_without_marker(status: int) -> None:
    info = classify_http_error(status, "invalid tool schema")
    assert info.kind is not LLMErrorKind.context_overflow
    assert not is_context_overflow(info)


def test_plain_400_is_bad_request() -> None:
    info = classify_http_error(400, "invalid tool schema")
    assert info.kind is LLMErrorKind.bad_request
    assert info.status == 400
    assert info.code is None
    assert not info.retryable


def test_empty_body_400_is_bad_request() -> None:
    info = classify_http_error(400, "")
    assert info.kind is LLMErrorKind.bad_request


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, LLMErrorKind.auth, False),
        (403, LLMErrorKind.auth, False),
        (404, LLMErrorKind.not_found, False),
        (408, LLMErrorKind.timeout, True),
        (429, LLMErrorKind.rate_limit, True),
        (500, LLMErrorKind.server, True),
        (503, LLMErrorKind.server, True),
    ],
)
def test_status_classification(status: int, kind: LLMErrorKind, retryable: bool) -> None:
    info = classify_http_error(status, '{"error": {"message": "boom"}}')
    assert info.kind is kind
    assert info.status == status
    assert info.retryable is retryable


@pytest.mark.parametrize("status", [402, 405, 409, 415])
def test_other_4xx_is_bad_request(status: int) -> None:
    info = classify_http_error(status, "no")
    assert info.kind is LLMErrorKind.bad_request
    assert not info.retryable


def test_timeout_exception() -> None:
    info = classify_exception(httpx.TimeoutException("timed out"))
    assert info.kind is LLMErrorKind.timeout
    assert info.retryable
    assert info.status is None
    assert info.message.startswith("TimeoutException:")


def test_connect_error_is_network() -> None:
    info = classify_exception(httpx.ConnectError("connection refused"))
    assert info.kind is LLMErrorKind.network
    assert info.retryable
    assert info.message.startswith("ConnectError:")


def test_generic_network_error() -> None:
    info = classify_exception(httpx.NetworkError("no route"))
    assert info.kind is LLMErrorKind.network
    assert info.retryable


def test_unknown_exception() -> None:
    info = classify_exception(ValueError("boom"))
    assert info.kind is LLMErrorKind.unknown
    assert not info.retryable
    assert info.status is None


def test_redaction_in_http_message() -> None:
    body = '{"error": {"message": "key sk-abcdefgh12345678 failed"}}'
    info = classify_http_error(401, body)
    assert "[REDACTED]" in info.message
    assert "sk-abcdefgh12345678" not in info.message


def test_redaction_in_exception_message() -> None:
    info = classify_exception(httpx.ConnectError("key sk-abcdefgh12345678 refused"))
    assert "[REDACTED]" in info.message
    assert "sk-abcdefgh12345678" not in info.message


def test_message_truncated_to_500() -> None:
    info = classify_http_error(500, "x" * 2000)
    assert len(info.message) == 500


def test_provider_error_info_is_frozen() -> None:
    info = classify_http_error(429, "limit")
    with pytest.raises(AttributeError):
        info.message = "changed"  # type: ignore[misc]
    assert isinstance(info, ProviderErrorInfo)


@pytest.mark.parametrize("status", [401, 500, 503])
def test_phrase_in_other_errors_is_not_overflow(status: int) -> None:
    info = classify_http_error(status, '{"error": {"message": "maximum context length is 8k"}}')
    assert not is_context_overflow(info)
