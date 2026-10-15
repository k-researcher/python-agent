"""Tests for provider authentication (agent.provider_auth)."""

from __future__ import annotations

import asyncio
import ssl
import time
import urllib.parse
from pathlib import Path

import httpx
import pytest
from httpx import Request, Response

from agent.provider_auth import (
    AuthError,
    AuthProvider,
    BearerAuth,
    CommandTokenAuth,
    HeaderKeyAuth,
    NoAuth,
    OAuthClientCredentials,
    build_ssl_context,
)

SECRET = "top-secret-key"
TOKEN = "tok-12345"
BASIC = "YmluYXJ5LWtleQ=="


def _run(coro):
    """Run a coroutine to completion."""
    return asyncio.run(coro)


def _http_handler(payload: dict, status: int = 200):
    """Build a MockTransport handler that returns a JSON body."""

    def handler(request: Request) -> Response:
        return Response(status, json=payload, request=request)

    return handler


def _provider(
    spec,
    handler,
    *,
    clock=None,
    refresh_margin: float = 60.0,
) -> AuthProvider:
    """Build an AuthProvider with a MockTransport client."""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, trust_env=False)
    return AuthProvider(
        spec,
        client=client,
        clock=clock or time.time,
        refresh_margin_seconds=refresh_margin,
    )


# --- static header specs ---------------------------------------------------


def test_header_key_auth() -> None:
    """HeaderKeyAuth sends the header with prefix and secret."""
    spec = HeaderKeyAuth(secret=SECRET)
    provider = AuthProvider(spec)
    result = _run(provider.headers())
    assert result == {"Authorization": "Api-Key " + SECRET}


def test_header_key_auth_custom_header_and_prefix() -> None:
    """HeaderKeyAuth honors a custom header name and prefix."""
    spec = HeaderKeyAuth(secret=SECRET, header="X-Key", prefix="K ")
    provider = AuthProvider(spec)
    result = _run(provider.headers())
    assert result == {"X-Key": "K " + SECRET}


def test_bearer_auth() -> None:
    """BearerAuth sends the bearer header."""
    spec = BearerAuth(secret=SECRET)
    provider = AuthProvider(spec)
    result = _run(provider.headers())
    assert result == {"Authorization": "Bearer " + SECRET}


def test_no_auth() -> None:
    """NoAuth sends no headers."""
    spec = NoAuth()
    provider = AuthProvider(spec)
    result = _run(provider.headers())
    assert result == {}


# --- OAuth client credentials ---------------------------------------------


def test_oauth_form_and_headers() -> None:
    """OAuth sends the expected form, Basic and Accept headers."""
    seen: dict = {}

    def handler(request: Request) -> Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["accept"] = request.headers.get("Accept")
        seen["content_type"] = request.headers.get("Content-Type")
        seen["rquid"] = request.headers.get("RqUID")
        # httpx encodes form data; read it from content.
        seen["form"] = {}
        for part in request.content.decode().split("&"):
            key, _, value = part.partition("=")
            if key:
                seen["form"][key] = urllib.parse.unquote_plus(value)
        return Response(
            200,
            json={"access_token": TOKEN, "expires_in": 3600},
            request=request,
        )

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="read write",
        extra_form={"grant_type": "client_credentials"},
    )
    provider = _provider(spec, handler)
    result = _run(provider.headers())

    assert result == {"Authorization": "Bearer " + TOKEN}
    assert seen["method"] == "POST"
    assert seen["url"] == "https://example.test/token"
    assert seen["authorization"] == "Basic " + BASIC
    assert seen["accept"] == "application/json"
    assert seen["content_type"] == "application/x-www-form-urlencoded"
    assert "rquid" in seen and seen["rquid"]
    assert seen["form"].get("scope") == "read write"
    assert seen["form"].get("grant_type") == "client_credentials"


def test_oauth_request_id_changes_each_request() -> None:
    """OAuth uses a new RqUID for every token request."""
    request_ids: list[str] = []

    def handler(request: Request) -> Response:
        request_ids.append(request.headers.get("RqUID", ""))
        return Response(200, json={"access_token": TOKEN}, request=request)

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
        expires_at_field=None,
        expires_in_field=None,
    )
    provider = _provider(spec, handler)
    # First call fetches a token.
    _run(provider.headers())
    provider.invalidate()
    _run(provider.headers())

    assert len(request_ids) == 2
    assert request_ids[0] != request_ids[1]
    assert all(request_ids)


def test_oauth_no_request_id_header() -> None:
    """OAuth skips the request id header when configured to None."""
    seen: dict = {}

    def handler(request: Request) -> Response:
        seen["rquid"] = request.headers.get("RqUID")
        return Response(200, json={"access_token": TOKEN}, request=request)

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
        request_id_header=None,
    )
    provider = _provider(spec, handler)
    _run(provider.headers())
    assert seen["rquid"] is None


def test_oauth_expires_at_ms() -> None:
    """OAuth honors an absolute expiry timestamp in milliseconds."""
    now = 1_700_000_000.0
    expiry_ms = int((now + 500.0) * 1000)

    def handler(request: Request) -> Response:
        return Response(
            200,
            json={"access_token": TOKEN, "expires_at": expiry_ms},
            request=request,
        )

    clock_state = {"now": now}

    def clock() -> float:
        return clock_state["now"]

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
    )
    provider = _provider(spec, handler, clock=clock)
    assert _run(provider.headers())["Authorization"] == "Bearer " + TOKEN
    # Within the cached window, no second fetch.
    calls = [0]

    def handler2(request: Request) -> Response:
        calls[0] += 1
        return Response(200, json={"access_token": "t2"}, request=request)

    provider2 = _provider(spec, handler2, clock=clock)
    assert _run(provider2.headers())["Authorization"] == "Bearer t2"
    assert calls[0] == 1
    # Still cached; no second fetch.
    assert _run(provider2.headers())["Authorization"] == "Bearer t2"
    assert calls[0] == 1


def test_oauth_expires_at_s() -> None:
    """OAuth honors an absolute expiry timestamp in seconds."""
    now = 1_700_000_000.0
    expiry_s = now + 500.0

    def handler(request: Request) -> Response:
        return Response(
            200,
            json={"access_token": TOKEN, "expires_at": expiry_s},
            request=request,
        )

    clock_state = {"now": now}

    def clock() -> float:
        return clock_state["now"]

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
        expires_at_unit="s",
    )
    provider = _provider(spec, handler, clock=clock)
    result = _run(provider.headers())
    assert result["Authorization"] == "Bearer " + TOKEN


def test_oauth_expires_in() -> None:
    """OAuth honors the expires_in field in seconds."""
    now = 1_700_000_000.0
    fetches = [0]

    def handler(request: Request) -> Response:
        fetches[0] += 1
        return Response(200, json={"access_token": TOKEN, "expires_in": 100}, request=request)

    clock_state = {"now": now}

    def clock() -> float:
        return clock_state["now"]

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
        expires_at_field=None,
        expires_in_field="expires_in",
    )
    provider = _provider(spec, handler, clock=clock)
    assert _run(provider.headers())["Authorization"] == "Bearer " + TOKEN
    assert fetches[0] == 1
    # Advance 20 seconds: 100 - 20 = 80 > 60 margin. Token is cached.
    clock_state["now"] = now + 20.0
    assert _run(provider.headers())["Authorization"] == "Bearer " + TOKEN
    assert fetches[0] == 1
    # Advance 70 seconds total: 100 - 70 = 30 < 60 margin. Refetch occurs.
    clock_state["now"] = now + 70.0
    assert _run(provider.headers())["Authorization"] == "Bearer " + TOKEN
    assert fetches[0] == 2


def test_oauth_refresh_before_expiry() -> None:
    """OAuth refetches when the token is close to expiry."""
    now = 1_700_000_000.0
    expiry = now + 100.0  # 100 seconds from now
    calls: list[str] = []

    def handler(request: Request) -> Response:
        calls.append(request.url.path)
        return Response(
            200,
            json={"access_token": TOKEN, "expires_at": expiry * 1000},
            request=request,
        )

    clock_state = {"now": now}

    def clock() -> float:
        return clock_state["now"]

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
    )
    provider = _provider(spec, handler, clock=clock)
    assert _run(provider.headers())["Authorization"] == "Bearer " + TOKEN
    # Advance 50 seconds: 100 - 50 = 50 < margin (60). A refetch occurs.
    clock_state["now"] = now + 50.0
    assert _run(provider.headers())["Authorization"] == "Bearer " + TOKEN
    assert len(calls) == 2


def test_oauth_ten_concurrent_headers_one_request() -> None:
    """Ten concurrent headers() calls share a single token fetch."""
    call_count = [0]

    def handler(request: Request) -> Response:
        call_count[0] += 1
        return Response(200, json={"access_token": TOKEN}, request=request)

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
        expires_at_field=None,
        expires_in_field="expires_in",
    )
    provider = _provider(spec, handler)

    async def run_many() -> list[dict[str, str]]:
        results = await asyncio.gather(*[provider.headers() for _ in range(10)])
        return results

    results = _run(run_many())
    assert all(r == {"Authorization": "Bearer " + TOKEN} for r in results)
    assert call_count[0] == 1


def test_oauth_invalidate_forces_new_request() -> None:
    """invalidate() forces a fresh token request."""
    call_count = [0]
    tokens_issued = ["first", "second"]

    def handler(request: Request) -> Response:
        idx = call_count[0]
        call_count[0] += 1
        return Response(200, json={"access_token": tokens_issued[idx]}, request=request)

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
        expires_at_field=None,
        expires_in_field=None,
    )
    provider = _provider(spec, handler)
    first = _run(provider.headers())
    second = _run(provider.headers())  # cached
    assert first == second == {"Authorization": "Bearer first"}
    assert call_count[0] == 1

    provider.invalidate()
    third = _run(provider.headers())
    assert third == {"Authorization": "Bearer second"}
    assert call_count[0] == 2


def test_oauth_401_no_key_leak() -> None:
    """OAuth 401 error never exposes the Basic key."""
    handler = _http_handler({"error": "invalid_client"}, status=401)
    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
    )
    provider = _provider(spec, handler)
    with pytest.raises(AuthError) as exc_info:
        _run(provider.headers())
    assert BASIC not in str(exc_info.value)
    assert "401" in str(exc_info.value)


def test_oauth_not_json_no_key_leak() -> None:
    """OAuth non-JSON error never exposes the Basic key."""

    def handler(request: Request) -> Response:
        return Response(200, text="not json", request=request)

    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
    )
    provider = _provider(spec, handler)
    with pytest.raises(AuthError) as exc_info:
        _run(provider.headers())
    assert BASIC not in str(exc_info.value)


def test_oauth_missing_token_field() -> None:
    """OAuth response without token_field raises AuthError."""
    handler = _http_handler({"foo": "bar"})
    spec = OAuthClientCredentials(
        token_url="https://example.test/token",
        basic_key=BASIC,
        scope="s",
    )
    provider = _provider(spec, handler)
    with pytest.raises(AuthError):
        _run(provider.headers())


# --- command token auth ----------------------------------------------------


def test_command_token_success() -> None:
    """CommandTokenAuth uses command standard output as a token."""

    import sys

    argv = (sys.executable, "-c", f"print('{TOKEN}')")
    spec = CommandTokenAuth(argv=argv, ttl_seconds=300.0)
    provider = AuthProvider(spec)
    result = _run(provider.headers())
    assert result == {"Authorization": "Bearer " + TOKEN}


def test_command_token_nonzero_exit() -> None:
    """CommandTokenAuth raises on a non-zero exit code."""
    import sys

    argv = (sys.executable, "-c", "import sys; sys.exit(3)")
    spec = CommandTokenAuth(argv=argv)
    provider = AuthProvider(spec)
    with pytest.raises(AuthError):
        _run(provider.headers())


def test_command_token_empty_output() -> None:
    """CommandTokenAuth raises on empty standard output."""
    import sys

    argv = (sys.executable, "-c", "print('   ')")
    spec = CommandTokenAuth(argv=argv)
    provider = AuthProvider(spec)
    with pytest.raises(AuthError):
        _run(provider.headers())


def test_command_token_cache_by_ttl() -> None:
    """CommandTokenAuth caches the token within its TTL."""
    import sys

    argv = (sys.executable, "-c", "print('tok-cmd')")
    spec = CommandTokenAuth(argv=argv, ttl_seconds=100.0)
    clock_state = {"now": 1000.0}

    def clock() -> float:
        return clock_state["now"]

    provider = AuthProvider(spec, clock=clock)
    first = _run(provider.headers())
    clock_state["now"] = 1050.0  # 50 s elapsed, well within TTL
    second = _run(provider.headers())
    assert first == second == {"Authorization": "Bearer tok-cmd"}


# --- build_ssl_context ------------------------------------------------------


def test_build_ssl_context_none() -> None:
    """build_ssl_context(None) returns True."""
    assert build_ssl_context(None) is True


def test_build_ssl_context_existing_file() -> None:
    """build_ssl_context returns a context for an existing CA file."""
    import datetime
    import tempfile
    from pathlib import Path as _Path

    import cryptography.hazmat.primitives.asymmetric.rsa as rsa
    import cryptography.hazmat.primitives.hashes as hashes
    import cryptography.hazmat.primitives.serialization as serialization
    import cryptography.x509 as x509
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.UTC)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    with tempfile.TemporaryDirectory() as tmpdir:
        ca_file = _Path(tmpdir) / "ca.pem"
        ca_file.write_bytes(pem)
        result = build_ssl_context(str(ca_file))
        assert isinstance(result, ssl.SSLContext)


def test_build_ssl_context_missing_file() -> None:
    """build_ssl_context raises AuthError for a missing CA file."""
    missing = str(Path("/nonexistent/ca-bundle.pem"))
    with pytest.raises(AuthError):
        build_ssl_context(missing)


# --- AuthProvider client ownership -----------------------------------------


def test_aclose_closes_own_client() -> None:
    """aclose() closes only the internally-created client."""
    spec = NoAuth()
    provider = AuthProvider(spec)
    assert provider._own_client is True
    _run(provider.aclose())


def test_aclose_keeps_caller_client() -> None:
    """aclose() does not close a caller-supplied client."""
    transport = httpx.MockTransport(_http_handler({}))
    client = httpx.AsyncClient(transport=transport, trust_env=False)
    spec = NoAuth()
    provider = AuthProvider(spec, client=client)
    _run(provider.aclose())
    assert not client.is_closed


def test_secrets_are_hidden_in_repr() -> None:
    from agent.provider_auth import BearerAuth, HeaderKeyAuth, OAuthClientCredentials

    secret = "secret-value-1234"
    specs = [
        HeaderKeyAuth(secret=secret),
        BearerAuth(secret=secret),
        OAuthClientCredentials(token_url="https://a.test/t", basic_key=secret, scope="s"),
    ]
    assert all(secret not in repr(spec) for spec in specs)


async def test_cancelled_command_fetch_stops_the_command(tmp_path: Path) -> None:
    import asyncio
    import os
    import sys

    from agent.provider_auth import AuthProvider, CommandTokenAuth

    pid_file = tmp_path / "token.pid"
    script = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    )
    provider = AuthProvider(CommandTokenAuth(argv=(sys.executable, "-c", script)))
    task = asyncio.create_task(provider.headers())
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await provider.aclose()
