from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
import typing
from urllib.parse import urlparse

import httpcore
import httpx

from agent.tools.base import ToolError


def _check_host(hostname: str, allowlist: list[str]) -> str:
    host = hostname.lower().rstrip(".")
    if not any(fnmatch.fnmatch(host, pattern.lower()) for pattern in allowlist):
        raise ToolError(f"Destination host is not in AGENT_NETWORK_ALLOWLIST: {host}")
    return host


def _blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True for every address that is not a public unicast address.

    An IPv4-mapped IPv6 address is checked as its IPv4 address. ``is_global`` is False
    for private, loopback, link-local, shared (CGNAT 100.64.0.0/10), documentation and
    other special ranges; the other flags close the remaining special cases.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


# Upper limit of one DNS lookup, also when the caller gives no timeout.
DNS_TIMEOUT_SECONDS = 10.0


async def resolve_checked(
    hostname: str, port: int, allowlist: list[str], *, timeout: float | None = None
) -> list[str]:
    """Resolve an allowed host and return its addresses; refuse when any address is blocked.

    The lookup stops after ``timeout`` seconds, and never later than DNS_TIMEOUT_SECONDS.
    """
    host = _check_host(hostname, allowlist)
    limit = DNS_TIMEOUT_SECONDS if timeout is None else min(timeout, DNS_TIMEOUT_SECONDS)
    try:
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM),
            timeout=max(limit, 0.0),
        )
    except TimeoutError as exc:
        raise ToolError(f"DNS lookup timed out: {host}") from exc
    except OSError as exc:
        raise ToolError(f"Cannot resolve destination host: {host}") from exc
    result: list[str] = []
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if _blocked(ip):
            raise ToolError(f"Destination resolves to a blocked address: {ip}")
        if str(ip) not in result:
            result.append(str(ip))
    if not result:
        raise ToolError(f"Cannot resolve destination host: {host}")
    return result


async def validate_outbound_url(url: str, allowlist: list[str]) -> str:
    """Check the scheme, the allowlist and the resolved addresses of a URL; return the host."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("Only absolute HTTP/HTTPS URLs are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    await resolve_checked(parsed.hostname, port, allowlist)
    return _check_host(parsed.hostname, allowlist)


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connect only to addresses that pass the outbound check at connect time.

    httpx normally resolves the host again after validate_outbound_url, so DNS can return
    a different, blocked address in between (DNS rebinding). This backend resolves and
    checks the host in connect_tcp and connects to a checked address of that same
    resolution. TLS still uses the original host name for SNI and certificate checks,
    because httpcore passes it to start_tls separately.
    """

    def __init__(
        self, allowlist: list[str], inner: httpcore.AsyncNetworkBackend | None = None
    ) -> None:
        self._allowlist = allowlist
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        loop = asyncio.get_running_loop()
        # One deadline covers the lookup and all connect attempts.
        deadline = None if timeout is None else loop.time() + timeout

        def remaining() -> float | None:
            return None if deadline is None else deadline - loop.time()

        addresses = await resolve_checked(host, port, self._allowlist, timeout=remaining())
        error: Exception | None = None
        for address in addresses:
            left = remaining()
            if left is not None and left <= 0:
                raise httpcore.ConnectTimeout(f"Connect deadline passed for {host}")
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=left,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                error = exc
        assert error is not None
        raise error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise ToolError("Unix socket connections are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class PinnedTransport(httpx.AsyncHTTPTransport):
    """An httpx transport whose connection pool uses PinnedNetworkBackend."""

    def __init__(
        self, allowlist: list[str], inner: httpcore.AsyncNetworkBackend | None = None
    ) -> None:
        # trust_env=False: SSL_CERT_FILE and SSL_CERT_DIR from the environment must not
        # change the trusted certificates. The default store and hostname checks stay on.
        super().__init__(trust_env=False)
        # httpx has no public option for the network backend: replace the pool it built.
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False),
            network_backend=PinnedNetworkBackend(allowlist, inner),
        )


def pinned_client(allowlist: list[str], *, timeout: float = 30.0) -> httpx.AsyncClient:
    """Return a client for outbound tool requests: checked addresses, no redirects, no proxy.

    trust_env is off: a proxy from the environment would make the connection bypass the
    address check.
    """
    return httpx.AsyncClient(
        transport=PinnedTransport(allowlist),
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    )
