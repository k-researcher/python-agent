from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx

from agent.inference_guard import CURRENT_GUARD, GuardContext, terminal_reason
from agent.llm_errors import (
    TERMINAL_GUARDS,
    LLMErrorKind,
    ProviderErrorInfo,
    classify_exception,
    classify_http_error,
)
from agent.llm_functions_dialect import from_functions_response, from_functions_stream_delta
from agent.model_config import SUPPORTED_TRANSPORTS, ReasoningTagsSpec, ResolvedModel
from agent.provider_auth import (
    AuthError,
    AuthProvider,
    AuthSpec,
    BearerAuth,
    CommandTokenAuth,
    HeaderKeyAuth,
    NoAuth,
    OAuthClientCredentials,
    build_ssl_context,
)
from agent.reasoning_tags import split_reasoning
from agent.redaction import redact
from agent.repetition import RepetitionDetector
from agent.response_validation import calls_validated, envelope, finish, usable, validate_functions
from agent.wire import WireRequest, body_sha256, build_wire_request

__all__ = ["LLMClient", "LLMError", "LLMResponse", "LLMTransientError", "redact"]

# Receives ("content" | "reasoning", text) for each streamed fragment.
DeltaCallback = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Audit one HTTP attempt inside a single send() call."""

    attempt: int
    status: Literal["started", "completed", "error"]
    payload_bytes: int
    payload_sha256: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_kind: str | None = None


@dataclass(slots=True)
class _AuditState:
    """Per-send() holder for the first audit failure on an error attempt record."""

    error: BaseException | None = None


class _CallbackError(Exception):
    """Marker: an exception raised by the ``on_delta`` callback.

    The attempt is already audited as ``error``/``unknown``; the wrapper only keeps the
    original object so the outer transport handlers never re-classify it as a provider
    failure, and the request is never retried.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__()
        self.original = original


AttemptCallback = Callable[[AttemptRecord], Awaitable[None]]

# Headers that a provider config may not set: they carry framing or the request type.
_RESERVED_HEADERS = frozenset(
    {"content-type", "content-length", "host", "transfer-encoding", "connection"}
)


class _AuthExpired(Exception):
    """Marker: the provider answered 401 before any fragment; the token may be stale."""

    def __init__(self, token: str | None, next_attempt: int) -> None:
        super().__init__()
        self.token = token
        self.next_attempt = next_attempt


def _auth_spec(model: ResolvedModel) -> AuthSpec:
    """Build the transport auth spec from the provider config and its resolved secret."""
    config = model.auth
    secret = model.auth_secret
    if config is None or config.type == "none":
        return NoAuth()
    if config.type == "header_key":
        return HeaderKeyAuth(secret=secret, header=config.header, prefix=config.prefix)
    if config.type == "bearer":
        return BearerAuth(secret=secret)
    if config.type == "oauth_client_credentials":
        return OAuthClientCredentials(
            token_url=config.token_url,
            basic_key=secret,
            scope=config.scope,
            request_id_header=config.request_id_header,
            expires_at_unit=config.expires_at_unit,
        )
    return CommandTokenAuth(argv=tuple(config.argv), ttl_seconds=config.ttl_seconds)


class LLMError(RuntimeError):
    """A failed LLM request. ``info`` holds the typed classification.

    ``started`` is True when the reply stream already sent content, reasoning or a tool call.
    After that, no retry, fallback or context retry is permitted.
    """

    def __init__(
        self, message: str, *, info: ProviderErrorInfo | None = None, started: bool = False
    ) -> None:
        super().__init__(message)
        self.info = info or ProviderErrorInfo(
            LLMErrorKind.unknown, None, None, message[:500], isinstance(self, LLMTransientError)
        )
        self.started = started

    @property
    def kind(self) -> LLMErrorKind:
        return self.info.kind


class LLMTransientError(LLMError):
    """The provider is temporarily unavailable. A fallback model can take the request."""


def error_for(info: ProviderErrorInfo, message: str, *, started: bool = False) -> LLMError:
    """Return the exception class that matches the classification."""
    error = LLMTransientError if info.retryable else LLMError
    return error(message, info=info, started=started)


@dataclass(slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    # None: the provider did not report usage. Unknown is not zero.
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning: str | None = None
    # Diagnostic fragments only; never eligible for tool execution.
    interrupted_tool_calls: list[dict[str, Any]] | None = None


class _DeferredResponseClose(httpx.AsyncByteStream):
    """Keep EOF separate from transport cleanup so response validation runs first."""

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self.stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        # The inference worker closes the original stream in its cleanup scope.
        pass


class LLMClient:
    """OpenAI-compatible /chat/completions client bound to one resolved model."""

    def __init__(self, model: ResolvedModel) -> None:
        self.model = model
        self._http: httpx.AsyncClient | None = None
        self._auth: AuthProvider | None = None

    @property
    def profile_name(self) -> str:
        return self.model.id

    @property
    def endpoint(self) -> str:
        return f"{self.model.base_url.rstrip('/')}/chat/completions"

    def _client(self) -> httpx.AsyncClient:
        # One pooled client per model keeps connections (and TLS sessions) alive between steps.
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.model.timeout_seconds, connect=15.0),
                follow_redirects=False,
                verify=self._verify(),
            )
        return self._http

    def _verify(self) -> ssl.SSLContext | bool:
        """Return a verifying TLS setting; an extra CA bundle adds to the system roots."""
        try:
            return build_ssl_context(self.model.ca_bundle)
        except AuthError as exc:
            raise LLMError(f"Model {self.model.id}: {exc}") from None

    def _auth_provider(self) -> AuthProvider | None:
        if self.model.auth is None:
            return None
        if self._auth is None:
            self._auth = AuthProvider(_auth_spec(self.model), verify=self._verify())
        return self._auth

    def _secrets(self, sent: str | None = None) -> tuple[str | None, ...]:
        """Return every secret that must never appear in an error text.

        ``sent`` is the token of the current attempt: a parallel request may already have
        replaced the cached token, but the error text can still quote the sent one.
        """
        cached = self._auth.token if self._auth is not None else None
        return (self.model.api_key, self.model.auth_secret, cached, sent)

    async def _headers(self) -> tuple[dict[str, str], str | None]:
        """Build the headers of one attempt and return the token they carry, if any."""
        auth_header = "authorization"
        if self.model.auth is not None and self.model.auth.type == "header_key":
            auth_header = self.model.auth.header.lower()
        for name in self.model.headers:
            if name.lower() in _RESERVED_HEADERS or name.lower() == auth_header:
                raise LLMError(f"Model {self.model.id}: provider headers must not set {name!r}")
        headers = {**self.model.headers, "Content-Type": "application/json"}
        provider = self._auth_provider()
        if provider is None:
            if self.model.api_key:
                headers["Authorization"] = f"Bearer {self.model.api_key}"
            return headers, None
        try:
            headers.update(await provider.headers())
        except AuthError as exc:
            message = redact(str(exc), tuple(item for item in self._secrets() if item))
            raise LLMError(f"Provider authentication failed: {message}") from None
        return headers, provider.token if provider.refreshable else None

    def _convert(self, data: Any) -> Any:
        """Turn a reply in the provider dialect into the chat-completions shape."""
        if self.model.dialect == "functions":
            validate_functions(data, stream=False)
            return from_functions_response(data)
        return data

    def _convert_chunk(self, data: str) -> str:
        if self.model.dialect != "functions" or data == "[DONE]":
            return data
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return data  # The accumulator reports the invalid chunk.
        validate_functions(parsed, stream=True)
        return json.dumps(from_functions_stream_delta(parsed), ensure_ascii=False)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._auth is not None:
            await self._auth.aclose()
            self._auth = None

    async def send(
        self,
        request: WireRequest,
        *,
        on_delta: DeltaCallback | None = None,
        on_attempt: AttemptCallback | None = None,
    ) -> LLMResponse:
        """Send one wire request and report each HTTP attempt to ``on_attempt``.

        The exact ``request.body`` bytes are transmitted. Retry 429, 5xx and transport
        errors, but never after the first streamed fragment has been delivered.
        """
        guard = CURRENT_GUARD.get()
        if guard is None:
            guard = GuardContext(self.model)
            return await guard.run(
                self.send(request, on_delta=on_delta, on_attempt=on_attempt), cleanup_worker=True
            )
        guard.ensure_open()
        guard.select(self.model)
        streaming = bool(request.payload.get("stream"))
        if streaming and on_delta is None:
            raise ValueError("on_delta is required when the request streams")
        if self.model.kind not in SUPPORTED_TRANSPORTS:
            raise LLMError(f"Provider type {self.model.kind} is not supported yet")
        if not self.model.configured:
            raise LLMError(f"Model {self.model.id} is not configured: API key is missing")

        sha256 = body_sha256(request)
        state = _AuditState()
        provider = self._auth_provider()
        # One token refresh after a 401, then at most one more request.
        refresh = provider is not None and provider.refreshable
        offset = 0
        while True:
            guard.ensure_open()
            try:
                if streaming:
                    assert on_delta is not None
                    return await self._stream(
                        state, request, sha256, on_delta, on_attempt, offset, refresh
                    )
                return await self._complete(state, request, sha256, on_attempt, offset, refresh)
            except _AuthExpired as expired:
                assert provider is not None
                provider.invalidate(expired.token)
                refresh = False
                offset = expired.next_attempt

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        """Build one wire request and send it. Stream when ``on_delta`` is set."""
        request = build_wire_request(
            self.model,
            messages,
            tools,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens or self.model.max_tokens,
            stream=on_delta is not None,
        )
        return await self.send(request, on_delta=on_delta)

    async def _complete(
        self,
        state: _AuditState,
        request: WireRequest,
        sha256: str,
        on_attempt: AttemptCallback | None,
        offset: int,
        refresh: bool,
    ) -> LLMResponse:
        """Send a non-streaming request. Retry 429 and 5xx with backoff.

        With ``refresh``, a 401 ends this call with ``_AuthExpired`` for one token refresh.
        """
        client = self._client()
        retries = self.model.max_retries
        payload_bytes = len(request.body)
        for attempt in range(retries + 1):
            guard = CURRENT_GUARD.get()
            assert guard is not None
            guard.ensure_open()
            number = attempt + offset
            observed_usage: Any = {}
            headers, token = await self._headers()
            await self._emit_attempt(state, on_attempt, number, "started", payload_bytes, sha256)
            try:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    content=request.body,
                    timeout=httpx.Timeout(
                        connect=client.timeout.connect,
                        write=client.timeout.write,
                        pool=client.timeout.pool,
                        read=600.0,
                    ),
                )
            except asyncio.CancelledError:
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind="cancelled",
                )
                raise
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                info = _redacted(classify_exception(exc), self._secrets(token))
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=info.kind,
                )
                if attempt >= retries:
                    raise error_for(info, f"LLM network error: {info.message}") from exc
                await asyncio.sleep(min(8.0, 2.0**attempt))
                continue
            except Exception as exc:
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=classify_exception(exc).kind,
                )
                raise
            if response.status_code == 401 and refresh:
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=LLMErrorKind.auth,
                )
                raise _AuthExpired(token, number + 1)
            try:
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < retries:
                    info = _redacted(
                        classify_http_error(response.status_code, response.text),
                        self._secrets(token),
                    )
                    await self._emit_attempt(
                        state,
                        on_attempt,
                        number,
                        "error",
                        payload_bytes,
                        sha256,
                        error_kind=info.kind,
                    )
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                    continue
                response.raise_for_status()
                body = response.json()
                observed_usage = body.get("usage", {}) if isinstance(body, dict) else {}
                guard = CURRENT_GUARD.get()
                if guard is not None:
                    guard.observe_usage(
                        token_count(observed_usage, "prompt_tokens"),
                        token_count(observed_usage, "completion_tokens"),
                    )
                parsed = self._parse_response(self._convert(body), self.model.reasoning.tags)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                info = _redacted(
                    classify_http_error(status, exc.response.text), self._secrets(token)
                )
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=info.kind,
                )
                raise error_for(info, f"LLM returned HTTP {status}: {info.message}") from exc
            except LLMError as exc:
                guard = CURRENT_GUARD.get()
                if guard is not None and exc.kind in TERMINAL_GUARDS:
                    guard.close(exc)
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=exc.kind,
                    prompt_tokens=token_count(observed_usage, "prompt_tokens"),
                    completion_tokens=token_count(observed_usage, "completion_tokens"),
                )
                raise
            except (KeyError, TypeError, ValueError) as exc:
                from agent.response_validation import invalid

                error = invalid()
                guard = CURRENT_GUARD.get()
                if guard is not None:
                    guard.close(error)
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=LLMErrorKind.invalid_response,
                    prompt_tokens=token_count(observed_usage, "prompt_tokens"),
                    completion_tokens=token_count(observed_usage, "completion_tokens"),
                )
                raise error from exc
            except Exception:
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=LLMErrorKind.unknown,
                )
                raise
            await self._emit_attempt(
                state,
                on_attempt,
                number,
                "completed",
                payload_bytes,
                sha256,
                prompt_tokens=parsed.prompt_tokens,
                completion_tokens=parsed.completion_tokens,
            )
            if state.error is not None:
                raise state.error
            return parsed
        raise LLMTransientError("LLM request failed")

    async def _stream(
        self,
        state: _AuditState,
        request: WireRequest,
        sha256: str,
        on_delta: DeltaCallback,
        on_attempt: AttemptCallback | None,
        offset: int,
        refresh: bool,
    ) -> LLMResponse:
        guard = CURRENT_GUARD.get()
        assert guard is not None
        progress = _SemanticProgress(guard)
        return await guard.idle(
            self._stream_work(
                state, request, sha256, on_delta, on_attempt, offset, refresh, progress
            ),
            self.model.semantic_idle_timeout_seconds,
            lambda: progress.at,
        )

    async def _stream_work(
        self,
        state: _AuditState,
        request: WireRequest,
        sha256: str,
        on_delta: DeltaCallback,
        on_attempt: AttemptCallback | None,
        offset: int,
        refresh: bool,
        progress: _SemanticProgress,
    ) -> LLMResponse:
        """Stream one request. A retry is possible only before the first received fragment.

        With ``refresh``, a 401 ends this call with ``_AuthExpired`` for one token refresh.
        """
        from agent.sse import ChatStreamAccumulator, SSEDecoder

        retries = self.model.max_retries
        payload_bytes = len(request.body)
        for attempt in range(retries + 1):
            guard = CURRENT_GUARD.get()
            assert guard is not None
            guard.ensure_open()
            number = attempt + offset
            started = False
            accumulator: ChatStreamAccumulator | None = None
            progress.reset()
            headers, token = await self._headers()
            await self._emit_attempt(state, on_attempt, number, "started", payload_bytes, sha256)
            try:
                manager = self._client().stream(
                    "POST", self.endpoint, headers=headers, content=request.body
                )
                response = await manager.__aenter__()
                transport_stream = response.stream
                assert isinstance(transport_stream, httpx.AsyncByteStream)
                response.stream = _DeferredResponseClose(transport_stream)

                async def close_response(
                    manager: AbstractAsyncContextManager[httpx.Response] = manager,
                    transport_stream: httpx.AsyncByteStream = transport_stream,
                ) -> None:
                    try:
                        await manager.__aexit__(None, None, None)
                    finally:
                        await transport_stream.aclose()

                try:
                    if response.status_code == 401 and refresh:
                        await response.aread()
                        await self._emit_attempt(
                            state,
                            on_attempt,
                            number,
                            "error",
                            payload_bytes,
                            sha256,
                            error_kind=LLMErrorKind.auth,
                        )
                        raise _AuthExpired(token, number + 1)
                    if response.status_code >= 400:
                        status = response.status_code
                        body = (await response.aread()).decode(errors="replace")
                        info = _redacted(classify_http_error(status, body), self._secrets(token))
                        if info.retryable and attempt < retries:
                            await self._emit_attempt(
                                state,
                                on_attempt,
                                number,
                                "error",
                                payload_bytes,
                                sha256,
                                error_kind=info.kind,
                            )
                            await asyncio.sleep(min(8.0, 2.0**attempt))
                            continue
                        raise error_for(info, f"LLM returned HTTP {status}: {info.message}")
                    decoder = SSEDecoder()
                    accumulator = ChatStreamAccumulator(
                        reasoning_tags=self.model.reasoning.tags, defer_text=True
                    )
                    detectors = {
                        kind: RepetitionDetector(self.model.repetition_threshold_chars)
                        for kind in ("content", "reasoning")
                    }

                    async def consume(
                        data: str | None,
                        accumulator: ChatStreamAccumulator = accumulator,
                        detectors: dict[str, RepetitionDetector] = detectors,
                        number: int = number,
                    ) -> None:
                        nonlocal started
                        await asyncio.sleep(0)
                        if data is None:
                            events = accumulator.end_events()
                        elif data != "[DONE]":
                            try:
                                raw = json.loads(data)
                            except ValueError:
                                from agent.response_validation import invalid

                                raise invalid() from None
                            if isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
                                progress.guard.observe_usage(
                                    token_count(raw["usage"], "prompt_tokens"),
                                    token_count(raw["usage"], "completion_tokens"),
                                )
                            await asyncio.sleep(0)
                            if self.model.dialect == "functions":
                                validate_functions(raw, stream=True)
                            progress.observe(raw)
                            payload = (
                                from_functions_stream_delta(raw)
                                if self.model.dialect == "functions"
                                else raw
                            )
                            await asyncio.sleep(0)
                            events = accumulator.add_payload(payload)
                        else:
                            events = accumulator.add(data)
                        await asyncio.sleep(0)
                        for event in events:
                            if event.kind == "tool_call":
                                started = True
                            elif event.kind in {"content", "reasoning"} and event.text:
                                started = True
                                if await detectors[event.kind].feed(event.text):
                                    from agent.inference_guard import guard_error

                                    raise guard_error(LLMErrorKind.repetition)
                                progress.guard.ensure_open()
                                accumulator.accept_text(event.kind, event.text)
                                await asyncio.sleep(0)
                                progress.guard.ensure_open()
                                try:
                                    await on_delta(event.kind, event.text)
                                except asyncio.CancelledError:
                                    raise
                                except BaseException as exc:
                                    await self._emit_attempt(
                                        state,
                                        on_attempt,
                                        number,
                                        "error",
                                        payload_bytes,
                                        sha256,
                                        error_kind=LLMErrorKind.unknown,
                                    )
                                    raise _CallbackError(exc) from exc
                                await asyncio.sleep(0)
                                progress.guard.ensure_open()

                    async for chunk in response.aiter_bytes():
                        for offset_bytes in range(0, len(chunk), 4096):
                            await asyncio.sleep(0)
                            for data in decoder.iter_feed(
                                chunk[offset_bytes : offset_bytes + 4096]
                            ):
                                await consume(data)
                                if accumulator.done:
                                    break
                            if accumulator.done:
                                break
                        if accumulator.done:
                            break
                    if not accumulator.done:
                        for data in decoder.close():
                            await consume(data)
                        await consume(None)
                    result = accumulator.finalize()
                except BaseException as error:
                    if isinstance(error, asyncio.CancelledError) or (
                        isinstance(error, LLMError) and error.kind in TERMINAL_GUARDS
                    ):
                        progress.guard.close(error)
                    raise
                finally:
                    guard = progress.guard
                    if guard.closed:
                        # A close failure is secondary to the first Stop or guard.
                        with suppress(Exception):
                            await guard.cleanup(close_response())
                    else:
                        await guard.finalize(close_response())
            except _CallbackError as exc:
                raise exc.original from None
            except _AuthExpired:
                raise
            except LLMError as exc:
                if exc.kind == LLMErrorKind.stream_incomplete and not started and attempt < retries:
                    await self._emit_attempt(
                        state,
                        on_attempt,
                        number,
                        "error",
                        payload_bytes,
                        sha256,
                        error_kind=exc.kind,
                    )
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                    continue
                # An error event inside the stream: keep its classification and state, but
                # never its secrets.
                secrets = tuple(item for item in self._secrets(token) if item)
                clean = type(exc)(
                    redact(str(exc), secrets),
                    info=_redacted(exc.info, self._secrets(token)),
                    started=exc.started or started,
                )
                counts = progress.guard.attempt_usage
                guard = progress.guard
                if exc.kind in TERMINAL_GUARDS:
                    guard.close(clean)
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=exc.kind,
                    prompt_tokens=counts[0],
                    completion_tokens=counts[1],
                )
                raise clean from None
            except asyncio.CancelledError:
                counts = progress.guard.attempt_usage
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind="cancelled",
                    prompt_tokens=counts[0],
                    completion_tokens=counts[1],
                )
                raise
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                info = _redacted(classify_exception(exc), self._secrets(token))
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=info.kind,
                )
                if started:
                    message = f"LLM stream was interrupted: {info.message}"
                    raise error_for(info, message, started=True) from exc
                if attempt >= retries:
                    raise error_for(info, f"LLM network error: {info.message}") from exc
                await asyncio.sleep(min(8.0, 2.0**attempt))
            except Exception as exc:
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "error",
                    payload_bytes,
                    sha256,
                    error_kind=classify_exception(exc).kind,
                )
                raise
            else:
                await self._emit_attempt(
                    state,
                    on_attempt,
                    number,
                    "completed",
                    payload_bytes,
                    sha256,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
                if state.error is not None:
                    raise state.error
                return result
        raise LLMTransientError("LLM request failed")

    async def _emit_attempt(
        self,
        state: _AuditState,
        on_attempt: AttemptCallback | None,
        attempt: int,
        status: Literal["started", "completed", "error"],
        payload_bytes: int,
        sha256: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        error_kind: str | None = None,
    ) -> None:
        """Notify the callback about one attempt event.

        Errors from ``on_attempt`` propagate on the success path. On an error path the
        LLM error wins, so a failing callback never hides it; the failure is kept and
        re-raised once the request has otherwise succeeded.
        """
        guard = CURRENT_GUARD.get()
        if guard is not None and status == "started":
            guard.ensure_open()
            guard.attempt_usage = (None, None)
        if on_attempt is None:
            return
        if guard is not None and guard.closed:
            status = "error"
            error_kind = terminal_reason(guard.reason)
        if guard is not None and status != "started":
            prompt_tokens = prompt_tokens if prompt_tokens is not None else guard.attempt_usage[0]
            completion_tokens = (
                completion_tokens if completion_tokens is not None else guard.attempt_usage[1]
            )
        record = AttemptRecord(
            attempt=attempt,
            status=status,
            payload_bytes=payload_bytes,
            payload_sha256=sha256,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_kind=error_kind,
        )
        try:
            if guard is not None and guard.closed:
                await guard.cleanup(on_attempt(record))
            else:
                await on_attempt(record)
        except Exception as exc:
            if status != "error":
                raise
            if state.error is None:
                state.error = exc

    @staticmethod
    def _parse_response(
        body: dict[str, Any], reasoning_tags: ReasoningTagsSpec | None = None
    ) -> LLMResponse:
        choice = envelope(body, stream=False)
        assert choice is not None
        reason = finish(choice.get("finish_reason")) or "stop"
        message = choice["message"]
        usage = body.get("usage", {})
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        content = message.get("content")
        if reasoning_tags is not None and isinstance(content, str):
            content, tagged = split_reasoning(content, reasoning_tags.start, reasoning_tags.end)
            content = content or None
            if tagged:
                reasoning = (reasoning if isinstance(reasoning, str) else "") + tagged
        calls = [] if reason == "length" else calls_validated(message.get("tool_calls"))
        if reason != "length":
            usable(content, calls, reason)
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=reason,
            prompt_tokens=token_count(usage, "prompt_tokens"),
            completion_tokens=token_count(usage, "completion_tokens"),
            reasoning=reasoning if isinstance(reasoning, str) else None,
            interrupted_tool_calls=message.get("tool_calls") if reason == "length" else None,
        )


def token_count(usage: Any, name: str) -> int | None:
    """Return a reported token count, or None when it is missing or not valid."""
    value = usage.get(name) if isinstance(usage, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _redacted(info: ProviderErrorInfo, secrets: tuple[str | None, ...]) -> ProviderErrorInfo:
    """Return the classification with the configured key removed from the message."""
    known = tuple(secret for secret in secrets if secret)
    return replace(info, message=redact(info.message, known))


def stream_error_info(error: Any) -> ProviderErrorInfo:
    """Classify an error event that arrives inside a successful SSE stream."""
    body = json.dumps({"error": error}, ensure_ascii=False, default=str)
    status = error.get("status") if isinstance(error, dict) else None
    return classify_http_error(status if isinstance(status, int) else 0, body)


class _SemanticProgress:
    def __init__(self, guard: GuardContext) -> None:
        self.guard = guard
        self.at = guard.clock.now()
        self.metadata: dict[tuple[int, str], str] = {}

    def reset(self) -> None:
        self.at = self.guard.clock.now()
        self.metadata.clear()
        self.guard.progressed.set()

    def observe(self, raw: Any) -> None:
        if not isinstance(raw, dict) or raw.get("error"):
            return
        choice = envelope(raw, stream=True, functions=True)
        if choice is None:
            return
        delta = choice.get("delta", {})
        changed = any(
            isinstance(delta.get(k), str) and delta[k].strip()
            for k in ("content", "reasoning", "reasoning_content")
        )
        calls = delta.get("tool_calls") or []
        if isinstance(delta.get("function_call"), dict):
            calls = [{"index": 0, "function": delta["function_call"]}]
        for call in calls:
            index = call.get("index", 0)
            fn = call.get("function") or {}
            for key, value in (("id", call.get("id")), ("name", fn.get("name"))):
                if isinstance(value, str) and value and self.metadata.get((index, key)) != value:
                    self.metadata[index, key] = value
                    changed = True
            args = fn.get("arguments")
            if isinstance(args, dict) or isinstance(args, str) and args:
                changed = True
        if changed:
            self.at = self.guard.clock.now()
            self.guard.progressed.set()
