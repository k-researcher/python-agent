from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
from urllib.parse import urlparse

from agent.tools.base import ToolError


async def validate_outbound_url(url: str, allowlist: list[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("Only absolute HTTP/HTTPS URLs are allowed")

    hostname = parsed.hostname.lower().rstrip(".")
    if not any(fnmatch.fnmatch(hostname, pattern.lower()) for pattern in allowlist):
        raise ToolError(f"Destination host is not in AGENT_NETWORK_ALLOWLIST: {hostname}")

    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ToolError(f"Cannot resolve destination host: {hostname}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            )
        ):
            raise ToolError(f"Destination resolves to a blocked address: {ip}")
    return hostname
