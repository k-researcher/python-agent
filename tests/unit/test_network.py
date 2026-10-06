import asyncio
import socket

import pytest

from agent.tools.base import ToolError
from agent.tools.network import validate_outbound_url


async def test_rejects_host_outside_allowlist() -> None:
    with pytest.raises(ToolError, match="ALLOWLIST"):
        await validate_outbound_url("https://example.com/data", ["api.example.com"])


async def test_rejects_private_address_after_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ToolError, match="blocked address"):
        await validate_outbound_url("https://api.example.com/data", ["api.example.com"])


async def test_accepts_allowlisted_public_address(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    assert (
        await validate_outbound_url("https://api.example.com/data", ["*.example.com"])
        == "api.example.com"
    )
