"""DNS rebinding protection: the address is checked again at connect time."""

from __future__ import annotations

import asyncio
import socket
import typing
from typing import Any

import httpcore
import httpx
import pytest

from agent.tools.base import ToolError
from agent.tools.network import (
    PinnedNetworkBackend,
    PinnedTransport,
    pinned_client,
    validate_outbound_url,
)

ALLOW = ["example.com"]


class FakeResolver:
    """Return the next answer of a list on each lookup (the last one repeats)."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def __call__(self, host: str, port: int, **_kwargs: Any) -> list[Any]:
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        family = socket.AF_INET6 if ":" in answer else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (answer, port))]


class RecordingBackend(httpcore.AsyncNetworkBackend):
    """Record the connect target and refuse, so that no real connection is made."""

    def __init__(self) -> None:
        self.targets: list[tuple[str, int]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.targets.append((host, port))
        raise httpcore.ConnectError("test backend: no connection")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise AssertionError("unix sockets must not be used")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> FakeResolver:
    fake = FakeResolver("93.184.216.34")

    async def getaddrinfo(_loop: Any, host: str, port: int, **kwargs: Any) -> list[Any]:
        return await fake(host, port, **kwargs)

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getaddrinfo", getaddrinfo)
    return fake


async def test_connects_to_the_checked_address(resolver: FakeResolver) -> None:
    inner = RecordingBackend()
    async with httpx.AsyncClient(transport=PinnedTransport(ALLOW, inner)) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("https://example.com/page")
    assert inner.targets == [("93.184.216.34", 443)]


async def test_rebinding_between_check_and_connect_is_refused(resolver: FakeResolver) -> None:
    # The first lookup (validate_outbound_url) sees a public address; the second lookup
    # (connect time) sees loopback. No connection to loopback may happen.
    resolver.answers = ["93.184.216.34", "127.0.0.1"]
    inner = RecordingBackend()
    await validate_outbound_url("http://example.com/", ALLOW)
    async with httpx.AsyncClient(transport=PinnedTransport(ALLOW, inner)) as client:
        with pytest.raises(ToolError, match="blocked address: 127.0.0.1"):
            await client.get("http://example.com/")
    assert inner.targets == []


@pytest.mark.parametrize("answer", ["::1", "::ffff:127.0.0.1", "169.254.169.254", "10.0.0.5"])
async def test_blocked_addresses_are_refused_at_connect(
    resolver: FakeResolver, answer: str
) -> None:
    resolver.answers = [answer]
    inner = RecordingBackend()
    backend = PinnedNetworkBackend(ALLOW, inner)
    with pytest.raises(ToolError, match="blocked address"):
        await backend.connect_tcp("example.com", 80)
    assert inner.targets == []


async def test_host_outside_allowlist_is_refused_at_connect(resolver: FakeResolver) -> None:
    inner = RecordingBackend()
    with pytest.raises(ToolError, match="not in AGENT_NETWORK_ALLOWLIST"):
        await PinnedNetworkBackend(ALLOW, inner).connect_tcp("evil.test", 80)
    assert inner.targets == [] and resolver.calls == 0


async def test_unix_socket_is_refused() -> None:
    with pytest.raises(ToolError):
        await PinnedNetworkBackend(ALLOW).connect_unix_socket("/var/run/docker.sock")


def test_pinned_client_ignores_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = pinned_client(ALLOW)
    assert client.trust_env is False
    assert client.follow_redirects is False
    assert isinstance(client._transport, PinnedTransport)
    assert not client._mounts  # no proxy mount from the environment


@pytest.mark.parametrize(
    ("answer", "blocked"),
    [
        ("100.64.0.1", True),  # shared address space (CGNAT), first address
        ("100.127.255.255", True),  # shared address space, last address
        ("100.63.255.255", False),  # just below the range
        ("100.128.0.0", False),  # just above the range
        ("::ffff:100.64.0.1", True),  # IPv4-mapped shared address
        ("192.0.2.10", True),  # documentation range
        ("198.18.0.1", True),  # benchmarking range
        ("8.8.8.8", False),
        ("2606:4700:4700::1111", False),
    ],
)
async def test_only_public_addresses_are_allowed(
    resolver: FakeResolver, answer: str, blocked: bool
) -> None:
    resolver.answers = [answer]
    inner = RecordingBackend()
    backend = PinnedNetworkBackend(ALLOW, inner)
    if blocked:
        with pytest.raises(ToolError, match="blocked address"):
            await backend.connect_tcp("example.com", 443)
        assert inner.targets == []
    else:
        with pytest.raises(httpcore.ConnectError):
            await backend.connect_tcp("example.com", 443)
        assert inner.targets == [(answer, 443)]


async def test_hanging_dns_lookup_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def hang(_loop: Any, *_args: Any, **_kwargs: Any) -> list[Any]:
        await asyncio.sleep(3600)
        return []

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getaddrinfo", hang)
    backend = PinnedNetworkBackend(ALLOW, RecordingBackend())
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(ToolError, match="DNS lookup timed out"):
        await backend.connect_tcp("example.com", 443, timeout=0.05)
    assert loop.time() - started < 1.0
    # The check before the request is bounded too.
    monkeypatch.setattr("agent.tools.network.DNS_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(ToolError, match="DNS lookup timed out"):
        await validate_outbound_url("https://example.com/", ALLOW)


class SlowFailingBackend(RecordingBackend):
    async def connect_tcp(
        self, host: str, port: int, timeout: float | None = None, **_: Any
    ) -> Any:
        self.targets.append((host, port))
        await asyncio.sleep(0.04)
        raise httpcore.ConnectError("refused")


async def test_one_deadline_covers_all_connect_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    many = [f"93.184.216.{n}" for n in range(1, 21)]

    async def answer_all(_loop: Any, host: str, port: int, **_kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in many]

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getaddrinfo", answer_all)
    inner = SlowFailingBackend()
    backend = PinnedNetworkBackend(ALLOW, inner)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(httpcore.ConnectTimeout):
        await backend.connect_tcp("example.com", 443, timeout=0.1)
    assert loop.time() - started < 0.5
    assert len(inner.targets) < len(many)


def test_certificate_environment_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    import ssl

    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/nonexistent")
    transport = PinnedTransport(ALLOW)  # must not read the broken environment
    context = transport._pool._ssl_context
    assert context is not None
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True
