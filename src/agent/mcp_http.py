"""Streamable HTTP transport of the Model Context Protocol.

Each JSON-RPC message is one HTTP POST to the MCP endpoint. The server answers a request
with one JSON object or with a stream of server-sent events that ends with the response.
The client speaks both eras of the protocol. The modern version has no handshake and no
session: every request carries its version in ``_meta`` and in headers. The legacy
versions start with the initialize handshake and may keep a session. In the automatic mode
the client asks ``server/discover`` first and falls back to the handshake when the server
does not know it, as the specification recommends. It follows no redirects and takes no
proxy from the environment.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from agent.mcp_client import CALL_TIMEOUT, CLIENT_INFO, MAX_MESSAGE_BYTES, McpClient, McpError

MAX_URL_CHARS = 2048
# Versions with the initialize handshake and this transport; an older server uses
# another HTTP transport.
SUPPORTED_VERSIONS = frozenset({"2025-03-26", "2025-06-18", "2025-11-25"})
MODERN_VERSION = "2026-07-28"
# "auto": ask the server; "modern": the version without a handshake; "legacy": the handshake.
HttpProtocol = Literal["auto", "modern", "legacy"]
# Errors that only a modern server sends: header mismatch, a missing client capability
# and an unsupported version.
_MODERN_ERRORS = frozenset({-32020, -32021, -32022})
# Failures without an answer of the server: they say nothing about its era.
_NO_ANSWER = frozenset({"timed out", "could not connect", "connection failed"})
_CANCEL_SECONDS = 5.0
# The client sets these headers itself.
RESERVED_HEADERS = frozenset(
    {
        "accept",
        "connection",
        "content-length",
        "content-type",
        "host",
        "last-event-id",
        "mcp-protocol-version",
        "mcp-session-id",
        "transfer-encoding",
    }
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MAX_SESSION_ID_CHARS = 1024
_MAX_ERROR_CHARS = 300
_CLOSE_SECONDS = 5.0
_METHOD_NOT_FOUND = -32601


def url_refusal(url: str) -> str | None:
    """Return why the agent must not connect to this MCP address, or None.

    Only HTTPS goes to other computers; plain HTTP is allowed to this computer only.
    A login in the address and the link-local addresses (the cloud metadata service)
    are refused.
    """
    if len(url) > MAX_URL_CHARS or any(char in url for char in "\x00\r\n\t "):
        return "Адрес слишком длинный или содержит пробелы"
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - a wrong port raises ValueError
    except ValueError:
        return "Неверный адрес"
    host = (parts.hostname or "").lower()
    if parts.scheme not in {"https", "http"} or not host:
        return "Адрес должен начинаться с https://"
    if parts.username is not None or parts.password is not None:
        return "Не указывайте логин и пароль в адресе: добавьте заголовок"
    if parts.fragment:
        return "Адрес не должен содержать #"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address is not None and (
        address.is_link_local or address.is_unspecified or address.is_multicast
    ):
        return "Это служебный адрес, к нему агент не подключается"
    local = host in _LOCAL_HOSTS or (address is not None and address.is_loopback)
    if parts.scheme == "http" and not local:
        return "Без шифрования можно подключаться только к этому компьютеру: используйте https://"
    return None


def _connection_error(error: httpx.HTTPError) -> McpError:
    # The address may hold a key in its query, so it never goes into the message.
    if isinstance(error, httpx.TimeoutException):
        return McpError(None, "timed out", kind="timeout")
    if isinstance(error, httpx.ConnectError):
        return McpError(None, "could not connect", kind="transport_error")
    return McpError(None, "connection failed", kind="transport_error")


class HttpMcpClient(McpClient):
    """Talk to an MCP server over the streamable HTTP transport."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        request_timeout: float = 30.0,
        call_timeout: float = CALL_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        protocol: HttpProtocol = "auto",
    ) -> None:
        """Bind to an MCP endpoint but do not connect.

        Raise ValueError when the address is not allowed. ``headers`` are sent with every
        request, for example an access token.
        """
        if url_refusal(url) is not None:
            raise ValueError("MCP address is not allowed")
        super().__init__(call_timeout=call_timeout)
        self._url = url
        self._headers = dict(headers or {})
        self._request_timeout = request_timeout
        self._transport = transport
        self._mode: HttpProtocol = protocol
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._protocol: str | None = None
        self._modern = False
        self._next_id = 1

    @property
    def protocol_version(self) -> str | None:
        """Return the protocol version in use after the start."""
        return self._protocol

    async def start(self) -> None:
        """Open the connection and agree on the protocol version."""
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            transport=self._transport,
            # Only the connection has its own limit; each request has its whole wait below.
            timeout=httpx.Timeout(None, connect=self._request_timeout),
            follow_redirects=False,
            trust_env=False,
        )
        try:
            if self._mode == "legacy":
                await self._initialize()
            elif self._mode == "modern":
                await self._discover()
            else:
                try:
                    await self._discover()
                except McpError as error:
                    if not _legacy_answer(error):
                        raise
                    await self._initialize()
        except BaseException:
            await self.close()
            raise

    async def _discover(self) -> None:
        """Ask a modern server for its versions, capabilities and name."""
        self._modern = True
        self._protocol = MODERN_VERSION
        try:
            result = await self._request_once("server/discover", {})
        except McpError:
            self._modern = False
            self._protocol = None
            raise
        versions = result.get("supportedVersions")
        if not isinstance(versions, list) or MODERN_VERSION not in versions:
            raise McpError(None, "unsupported protocol version: the server offers another one")
        meta = result.get("_meta")
        info = meta.get("io.modelcontextprotocol/serverInfo") if isinstance(meta, dict) else None
        self.server_info = dict(info) if isinstance(info, dict) else {}
        capabilities = result.get("capabilities")
        self.capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}

    async def close(self) -> None:
        """End the server session and release the connection."""
        closing = asyncio.create_task(self._close())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        client = self._client
        if client is None:
            return
        headers = self._message_headers()
        self._client = None
        self._session_id = None
        self._protocol = None
        self._modern = False
        try:
            if "Mcp-Session-Id" in headers:
                # A server may refuse to end sessions (405); the session then expires.
                with contextlib.suppress(httpx.HTTPError, TimeoutError):
                    async with asyncio.timeout(_CLOSE_SECONDS):
                        await client.delete(self._url, headers=headers)
        finally:
            await client.aclose()

    def _message_headers(self, message: dict[str, Any] | None = None) -> dict[str, str]:
        headers = {
            **self._headers,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._protocol is not None:
            headers["MCP-Protocol-Version"] = self._protocol
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
        if self._modern and message is not None and isinstance(message.get("method"), str):
            # A modern server checks that these headers repeat the body.
            headers["Mcp-Method"] = message["method"]
            params = message.get("params")
            name = params.get("name", params.get("uri")) if isinstance(params, dict) else None
            if isinstance(name, str):
                headers["Mcp-Name"] = header_value(name)
        return headers

    async def _request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        try:
            return await self._request_once(method, params, timeout)
        except McpError as error:
            if error.status != 404 or method == "initialize" or self._session_id is None:
                raise
        # The server ended the session; the request was not handled, so a new session
        # may send it again.
        self._session_id = None
        self._protocol = None
        await self._initialize()
        return await self._request_once(method, params, timeout)

    async def _request_once(
        self, method: str, params: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        if self._client is None:
            raise McpError(None, "client is not started", kind="transport_error")
        call_id = self._next_id
        self._next_id += 1
        if self._modern:
            params = {**params, "_meta": {**params.get("_meta", {}), **_request_meta()}}
        message = {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params}
        try:
            async with asyncio.timeout(timeout or self._request_timeout):
                result = await self._send(message, call_id)
        except TimeoutError:
            await self._cancel_remote(method, call_id)
            raise McpError(None, "timed out", kind="timeout") from None
        assert result is not None
        result_type = result.get("resultType", "complete")
        if result_type == "input_required":
            # The client offers no sampling, elicitation or roots.
            raise McpError(None, "the server asks for input that the agent cannot give")
        if result_type != "complete":
            raise McpError(None, "protocol error: unknown result type")
        if method == "initialize":
            version = result.get("protocolVersion")
            if version not in SUPPORTED_VERSIONS:
                raise McpError(None, f"unsupported protocol version: {str(version)[:40]}")
            self._protocol = str(version)
        return result

    async def _cancel_remote(self, method: str, call_id: int) -> None:
        """Tell a legacy server that the result of a timed out request is no longer wanted.

        A modern server takes the closed response stream as the cancellation. A legacy
        server does not, so it gets the notification. The handshake is never cancelled.
        """
        if self._modern or method == "initialize":
            return
        message = {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": call_id, "reason": "timed out"},
        }
        with contextlib.suppress(McpError, httpx.HTTPError, TimeoutError):
            async with asyncio.timeout(_CANCEL_SECONDS):
                await self._send(message, None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self._client is None:
            raise McpError(None, "client is not started", kind="transport_error")
        if self._modern:
            # A modern server keeps no connection state to notify about.
            return
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            async with asyncio.timeout(self._request_timeout):
                await self._send(message, None)
        except TimeoutError:
            raise McpError(None, "timed out", kind="timeout") from None

    async def _send(self, message: dict[str, Any], call_id: int | None) -> dict[str, Any] | None:
        """Post one message; return the result of a request, or None for a notification."""
        assert self._client is not None
        body = json.dumps(message, separators=(",", ":"))
        try:
            async with self._client.stream(
                "POST", self._url, content=body, headers=self._message_headers(message)
            ) as response:
                await self._check_status(response)
                if call_id is None:
                    return None
                if response.status_code != 200:
                    raise McpError(None, "protocol error: no response to a request")
                if message.get("method") == "initialize":
                    self._keep_session(response)
                kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind == "application/json":
                    return self._result(await self._read_body(response), call_id)
                if kind == "text/event-stream":
                    return await self._read_events(response, call_id)
                raise McpError(None, "protocol error: unexpected content type")
        except httpx.HTTPError as error:
            raise _connection_error(error) from None

    def _keep_session(self, response: httpx.Response) -> None:
        session_id = response.headers.get("mcp-session-id")
        if session_id is None:
            return
        if (
            not session_id
            or len(session_id) > _MAX_SESSION_ID_CHARS
            or any(not 0x21 <= ord(char) <= 0x7E for char in session_id)
        ):
            raise McpError(None, "protocol error: invalid session ID")
        self._session_id = session_id

    async def _check_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status in {200, 202}:
            return
        if 300 <= status < 400:
            raise McpError(None, "the server redirects the request", status=status)
        if status in {401, 403}:
            raise McpError(None, "access denied", status=status)
        if status == 404 and self._session_id is not None:
            raise McpError(None, "session expired", status=status)
        detail = ""
        code: int | None = None
        with contextlib.suppress(McpError, ValueError, AttributeError, httpx.HTTPError):
            error = json.loads(await self._read_body(response, limit=64 * 1024)).get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), int):
                code = error["code"]
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                detail = ": " + " ".join(error["message"].split())[:_MAX_ERROR_CHARS]
        raise McpError(code, f"HTTP {status}{detail}", status=status)

    @staticmethod
    async def _read_body(response: httpx.Response, limit: int | None = None) -> bytes:
        limit = MAX_MESSAGE_BYTES if limit is None else limit
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > limit:
                raise McpError(None, "protocol error: message too large")
        return bytes(body)

    @staticmethod
    def _result(raw: bytes | dict[str, Any], call_id: int) -> dict[str, Any]:
        try:
            message = json.loads(raw) if isinstance(raw, bytes) else raw
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise McpError(None, "protocol error: response is not JSON") from None
        if not isinstance(message, dict) or message.get("id") != call_id:
            raise McpError(None, "protocol error: unexpected response")
        error = message.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, dict) else None
            text = error.get("message", "error") if isinstance(error, dict) else "error"
            raise McpError(code if isinstance(code, int) else None, str(text))
        result = message.get("result", {})
        if not isinstance(result, dict):
            raise McpError(None, "protocol error: result is not an object")
        return result

    async def _read_events(self, response: httpx.Response, call_id: int) -> dict[str, Any]:
        """Read server-sent events until the response to the request arrives."""
        buffer = bytearray()
        data: list[str] = []
        size = 0
        async for chunk in response.aiter_bytes():
            buffer.extend(chunk)
            for line in _complete_lines(buffer):
                if line:
                    field, _, value = line.partition(":")
                    if field == "data":
                        value = value[1:] if value.startswith(" ") else value
                        data.append(value)
                        size += len(value)
                        if size > MAX_MESSAGE_BYTES:
                            raise McpError(None, "protocol error: message too large")
                    # Comments, event names, IDs and retry times need no action.
                    continue
                found = await self._dispatch("\n".join(data), call_id)
                data, size = [], 0
                if found is not None:
                    return found
            if len(buffer) > MAX_MESSAGE_BYTES:
                raise McpError(None, "protocol error: line too long")
        if data:
            found = await self._dispatch("\n".join(data), call_id)
            if found is not None:
                return found
        raise McpError(None, "server closed the stream without a response", kind="transport_error")

    async def _dispatch(self, text: str, call_id: int) -> dict[str, Any] | None:
        if not text.strip():
            return None
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(message, dict):
            return None
        if isinstance(message.get("method"), str):
            if "id" in message:
                # A request of the server, such as sampling: this client offers none.
                reply = {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": _METHOD_NOT_FOUND, "message": "Method not found"},
                }
                with contextlib.suppress(McpError):
                    await self._send(reply, None)
            return None
        if message.get("id") != call_id:
            return None
        return self._result(message, call_id)


def _request_meta() -> dict[str, Any]:
    return {
        "io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
        "io.modelcontextprotocol/clientInfo": dict(CLIENT_INFO),
        "io.modelcontextprotocol/clientCapabilities": {},
    }


def header_value(value: str) -> str:
    """Return a header value; one that is not plain ASCII goes in the Base64 sentinel form."""
    plain = (
        value == value.strip()
        and all(0x20 <= ord(char) <= 0x7E for char in value)
        and not (value.startswith("=?base64?") and value.endswith("?="))
    )
    if plain:
        return value
    return "=?base64?" + base64.b64encode(value.encode()).decode() + "?="


def _legacy_answer(error: McpError) -> bool:
    """Tell whether a failed discovery means a server with the handshake.

    A modern server answers with a modern error; a refused access, a redirect or no answer
    at all says nothing about the era, so the client does not fall back then.
    """
    if error.code in _MODERN_ERRORS or error.message in _NO_ANSWER:
        return False
    return error.status not in {401, 403} and not (
        error.status is not None and 300 <= error.status < 400
    )


def _complete_lines(buffer: bytearray) -> list[str]:
    """Take the complete lines out of the buffer; a line ends with CRLF, LF or CR."""
    lines: list[str] = []
    start = 0
    index = 0
    while index < len(buffer):
        byte = buffer[index]
        if byte == 0x0A or byte == 0x0D:
            if byte == 0x0D and index + 1 == len(buffer):
                # A CR at the end may be the first half of CRLF.
                break
            lines.append(buffer[start:index].decode("utf-8", errors="replace"))
            index += 2 if byte == 0x0D and buffer[index + 1] == 0x0A else 1
            start = index
            continue
        index += 1
    del buffer[:start]
    return lines
