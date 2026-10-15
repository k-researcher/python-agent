"""Authentication specs for model providers.

This module builds HTTP request headers for several provider auth schemes.
It never connects to any model client. The module keeps all secrets out of
error messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

REDACTED = "***"
DEFAULT_TTL_SECONDS = 300.0
DEFAULT_REFRESH_MARGIN = 60.0
COMMAND_TIMEOUT = 30.0


class AuthError(RuntimeError):
    """Authentication failed.

    Never include a key or token in the message.
    """


def _redact(message: str, secrets: list[str]) -> str:
    """Replace every secret with ``***`` in a message."""
    for secret in secrets:
        if secret:
            message = message.replace(secret, REDACTED)
    return message


@dataclass(frozen=True, slots=True)
class HeaderKeyAuth:
    """Send a fixed header with a prefixed secret."""

    secret: str = field(repr=False)
    header: str = "Authorization"
    prefix: str = "Api-Key "


@dataclass(frozen=True, slots=True)
class BearerAuth:
    """Send the secret as a bearer token."""

    secret: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class OAuthClientCredentials:
    """Fetch an access token with the client credentials flow."""

    token_url: str
    basic_key: str = field(repr=False)
    scope: str
    request_id_header: str | None = "RqUID"
    token_field: str = "access_token"
    expires_at_field: str | None = "expires_at"
    expires_at_unit: str = "ms"  # "ms" or "s"
    expires_in_field: str | None = "expires_in"
    extra_form: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expires_at_unit not in {"ms", "s"}:
            raise ValueError(f"expires_at_unit must be 'ms' or 's': {self.expires_at_unit!r}")


@dataclass(frozen=True, slots=True)
class CommandTokenAuth:
    """Read a token from command standard output.

    The argv tuple is used directly. The shell is never used.
    """

    argv: tuple[str, ...]
    ttl_seconds: float = DEFAULT_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class NoAuth:
    """Send no authentication headers."""


AuthSpec = HeaderKeyAuth | BearerAuth | OAuthClientCredentials | CommandTokenAuth | NoAuth

Clock = Callable[[], float]


def build_ssl_context(ca_bundle: str | None) -> ssl.SSLContext | bool:
    """Build an SSL context for an extra CA bundle file.

    Return ``True`` when no bundle is given. That keeps the default TLS
    verification. A missing bundle raises :class:`AuthError`. TLS
    verification is never disabled.
    """
    if ca_bundle is None:
        return True
    from pathlib import Path

    if not Path(ca_bundle).is_file():
        raise AuthError("CA bundle file was not found")
    context = ssl.create_default_context()
    try:
        context.load_verify_locations(cafile=ca_bundle)
    except OSError:
        raise AuthError("CA bundle file could not be read") from None
    return context


class _CachedToken:
    """Hold a token and its expiry time.

    The expiry is a wall-clock timestamp in seconds. A value of ``None``
    means the token never expires.
    """

    __slots__ = ("token", "expires_at")

    def __init__(self, token: str, expires_at: float | None) -> None:
        self.token = token
        self.expires_at = expires_at


class AuthProvider:
    """Build authentication headers for one provider spec.

    Static specs resolve to fixed headers. OAuth and command specs use a
    cached token. Concurrent calls share one token fetch through a lock.
    """

    def __init__(
        self,
        spec: AuthSpec,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock = time.time,
        refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN,
        verify: ssl.SSLContext | bool = True,
    ) -> None:
        self._spec = spec
        self._clock = clock
        self._refresh_margin = refresh_margin_seconds
        self._own_client = client is None
        # The token endpoint can use the same national certificate as the provider.
        self._client = client or httpx.AsyncClient(trust_env=False, verify=verify)
        self._lock = asyncio.Lock()
        self._cached: _CachedToken | None = None

    @property
    def spec(self) -> AuthSpec:
        """Return the underlying auth spec."""
        return self._spec

    async def headers(self) -> dict[str, str]:
        """Return the HTTP headers for the current token."""
        spec = self._spec
        if isinstance(spec, HeaderKeyAuth):
            return {spec.header: spec.prefix + spec.secret}
        if isinstance(spec, BearerAuth):
            return {"Authorization": "Bearer " + spec.secret}
        if isinstance(spec, NoAuth):
            return {}
        token = await self._token()
        return {"Authorization": "Bearer " + token}

    @property
    def refreshable(self) -> bool:
        """Tell whether a new token can replace a rejected one (OAuth or command)."""
        return isinstance(self._spec, (OAuthClientCredentials, CommandTokenAuth))

    @property
    def token(self) -> str | None:
        """Return the cached token, so that callers can remove it from error texts."""
        return self._cached.token if self._cached is not None else None

    def invalidate(self, token: str | None = None) -> None:
        """Drop the cached token after a 401.

        With ``token``, drop the cache only while it still holds that token: a parallel
        request may already have fetched a new one, and that one must stay.
        """
        if token is None or (self._cached is not None and self._cached.token == token):
            self._cached = None

    async def aclose(self) -> None:
        """Close the owned HTTP client.

        A caller-supplied client is left open.
        """
        if self._own_client:
            await self._client.aclose()

    async def _token(self) -> str:
        """Return a fresh or still-valid token."""
        cached = self._cached
        now = self._clock()
        if cached is not None:
            if cached.expires_at is None:
                return cached.token
            if cached.expires_at - now > self._refresh_margin:
                return cached.token
        async with self._lock:
            cached = self._cached
            first = self._clock()
            if cached is not None:
                if cached.expires_at is None:
                    return cached.token
                if cached.expires_at - first > self._refresh_margin:
                    return cached.token
            token, expires_at = await self._fetch()
            self._cached = _CachedToken(token, expires_at)
            return token

    async def _fetch(self) -> tuple[str, float | None]:
        """Fetch a token and compute its expiry."""
        spec = self._spec
        if isinstance(spec, CommandTokenAuth):
            return await self._fetch_command(spec)
        if isinstance(spec, OAuthClientCredentials):
            return await self._fetch_oauth(spec)
        raise TypeError(f"Unsupported auth spec: {type(spec).__name__}")

    async def _fetch_command(self, spec: CommandTokenAuth) -> tuple[str, float | None]:
        """Run the command and return its standard output as a token."""
        process = await asyncio.create_subprocess_exec(
            *spec.argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            result = await asyncio.wait_for(process.communicate(), timeout=COMMAND_TIMEOUT)
        except TimeoutError:
            await _stop(process)
            raise AuthError("Command token fetch timed out") from None
        except BaseException:
            # A cancelled fetch must not leave the command running.
            await _stop(process)
            raise
        stdout, _stderr = result
        if process.returncode != 0:
            raise AuthError("Command token fetch failed")
        token = stdout.decode("utf-8", errors="replace").strip()
        if not token:
            raise AuthError("Command token fetch returned no token")
        return token, self._clock() + spec.ttl_seconds

    async def _fetch_oauth(self, spec: OAuthClientCredentials) -> tuple[str, float | None]:
        """Fetch an access token from the token endpoint."""
        headers: dict[str, str] = {
            "Authorization": "Basic " + spec.basic_key,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if spec.request_id_header is not None:
            request_id = str(uuid.uuid4())
            headers[spec.request_id_header] = request_id
        form = {"scope": spec.scope}
        form.update(spec.extra_form)
        try:
            response = await self._client.post(spec.token_url, headers=headers, data=form)
        except httpx.HTTPError as exc:
            raise AuthError(_redact(f"Token request failed: {exc}", [spec.basic_key])) from exc
        if response.status_code == 401:
            raise AuthError("Token request was unauthorized (401)")
        if response.status_code == 403:
            raise AuthError("Token request was forbidden (403)")
        if response.is_error:
            raise AuthError(f"Token request failed with status {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise AuthError("Token response was not valid JSON") from None
        if not isinstance(data, dict):
            raise AuthError("Token response was not a JSON object")
        token = data.get(spec.token_field)
        if not isinstance(token, str) or not token:
            raise AuthError("Token response did not include a token")
        expires_at = self._expiry(spec, data)
        return token, expires_at

    def _expiry(self, spec: OAuthClientCredentials, data: dict[str, Any]) -> float | None:
        """Compute the token expiry from the response fields."""
        now = self._clock()
        if spec.expires_at_field is not None:
            raw = data.get(spec.expires_at_field)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                value = float(raw)
                if spec.expires_at_unit == "ms":
                    value = value / 1000.0
                return value
        if spec.expires_in_field is not None:
            raw = data.get(spec.expires_in_field)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return now + float(raw)
        return now + DEFAULT_TTL_SECONDS


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Kill a command that is still running and reap it."""
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    await process.wait()
