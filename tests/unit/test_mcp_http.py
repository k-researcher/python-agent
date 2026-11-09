"""The MCP client over streamable HTTP, against a scripted server."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from agent.mcp_client import McpError
from agent.mcp_http import MODERN_VERSION, HttpMcpClient, header_value, url_refusal

URL = "https://mcp.example.com/mcp"
SESSION = "session-1"


@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectError, "transport_error"),
        (httpx.ReadError, "transport_error"),
    ],
)
async def test_connection_failures_have_safe_typed_categories(
    exception: type[httpx.HTTPError], kind: str
) -> None:
    requests = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise exception("credential-canary", request=request)

    client = HttpMcpClient(URL, transport=httpx.MockTransport(fail), protocol="modern")
    with pytest.raises(McpError) as caught:
        await client.start()
    assert caught.value.kind == kind
    assert "credential-canary" not in str(caught.value)
    assert len(requests) == 1


def rpc(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def reply(message: dict[str, Any], result: dict[str, Any], **headers: str) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": message["id"], "result": result}
    return httpx.Response(200, json=body, headers=headers)


def events(*items: str, **headers: str) -> httpx.Response:
    return httpx.Response(
        200,
        content="".join(items).encode(),
        headers={"content-type": "text/event-stream", **headers},
    )


def decoded(value: str) -> str:
    """Read a header value the way a modern server does."""
    if value.startswith("=?base64?") and value.endswith("?="):
        return base64.b64decode(value[9:-2]).decode()
    return value


def refusal(status: int, code: int, text: str, **data: Any) -> httpx.Response:
    error: dict[str, Any] = {"code": code, "message": text, **({"data": data} if data else {})}
    return httpx.Response(status, json={"jsonrpc": "2.0", "id": None, "error": error})


class Server:
    """A scripted MCP server: it keeps every request it gets.

    A legacy server needs the handshake; a modern one needs the version in every request.
    """

    def __init__(
        self,
        tools: Callable[[httpx.Request], httpx.Response] | None = None,
        *,
        modern: bool = False,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.tools = tools
        self.session = SESSION
        self.modern = modern

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200)
        message = rpc(request)
        method = message.get("method")
        if self.modern:
            return self.answer_modern(request, message)
        if method != "initialize" and request.headers.get("mcp-session-id") != self.session:
            # What the common SDKs answer before the handshake.
            return refusal(400, -32000, "Bad Request: No valid session ID provided")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "docs", "version": "1.0"},
            }
            return reply(message, result, **{"mcp-session-id": self.session})
        if method is not None and "id" not in message:
            return httpx.Response(202)
        if method == "tools/list" and self.tools is not None:
            return self.tools(request)
        if method == "tools/list":
            return reply(message, {"tools": [{"name": "search", "inputSchema": {}}]})
        if method == "tools/call":
            return reply(message, {"content": [{"type": "text", "text": "found"}]})
        return httpx.Response(202)

    def answer_modern(self, request: httpx.Request, message: dict[str, Any]) -> httpx.Response:
        method = message.get("method")
        params = message.get("params", {})
        meta = params.get("_meta", {})
        if meta.get("io.modelcontextprotocol/protocolVersion") != MODERN_VERSION:
            return refusal(400, -32602, "Invalid params: no protocol version")
        if (
            request.headers.get("mcp-protocol-version") != MODERN_VERSION
            or request.headers.get("mcp-method") != method
            or ("name" in params and decoded(request.headers.get("mcp-name", "")) != params["name"])
        ):
            return refusal(400, -32020, "Header mismatch")
        if method == "server/discover":
            return reply(
                message,
                {
                    "resultType": "complete",
                    "supportedVersions": [MODERN_VERSION],
                    "capabilities": {"tools": {}},
                    "_meta": {
                        "io.modelcontextprotocol/serverInfo": {"name": "new", "version": "2"}
                    },
                },
            )
        if method == "tools/list" and self.tools is not None:
            return self.tools(request)
        if method == "tools/list":
            return reply(message, {"resultType": "complete", "tools": [{"name": "поиск"}]})
        if method == "tools/call":
            return reply(message, {"resultType": "complete", "content": []})
        return refusal(404, -32601, "Method not found")

    def posts(self) -> list[dict[str, Any]]:
        return [rpc(request) for request in self.requests if request.method == "POST"]


def client(server: Server, **values: Any) -> HttpMcpClient:
    return HttpMcpClient(URL, transport=httpx.MockTransport(server), **values)


async def test_handshake_session_and_headers_on_every_request() -> None:
    server = Server()
    async with client(server, headers={"Authorization": "Bearer key"}, protocol="legacy") as mcp:
        tools = await mcp.list_tools()
        assert [tool.name for tool in tools] == ["search"]
        assert mcp.server_info == {"name": "docs", "version": "1.0"}
    first, initialized, listed, closed = server.requests
    assert rpc(first)["method"] == "initialize"
    assert "mcp-session-id" not in first.headers
    assert "mcp-protocol-version" not in first.headers
    for request in (initialized, listed, closed):
        assert request.headers["mcp-session-id"] == SESSION
        assert request.headers["mcp-protocol-version"] == "2025-06-18"
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer key"
    assert "text/event-stream" in listed.headers["accept"]
    assert "application/json" in listed.headers["accept"]
    assert rpc(initialized) == {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    # Closing ends the session on the server.
    assert closed.method == "DELETE"


async def test_automatic_mode_falls_back_to_the_handshake_of_a_legacy_server() -> None:
    server = Server()
    async with client(server) as mcp:
        assert [tool.name for tool in await mcp.list_tools()] == ["search"]
        assert mcp.protocol_version == "2025-06-18"
    methods = [message.get("method") for message in server.posts()]
    assert methods == ["server/discover", "initialize", "notifications/initialized", "tools/list"]


async def test_automatic_mode_speaks_the_modern_version_without_a_session() -> None:
    server = Server(modern=True)
    async with client(server, headers={"Authorization": "Bearer key"}) as mcp:
        assert mcp.protocol_version == MODERN_VERSION
        assert mcp.server_info == {"name": "new", "version": "2"}
        assert [tool.name for tool in await mcp.list_tools()] == ["поиск"]
        await mcp.call_tool("поиск", {"q": "x"})
    methods = [message.get("method") for message in server.posts()]
    assert methods == ["server/discover", "tools/list", "tools/call"]
    # No handshake, no session and nothing to end on close.
    assert all(request.method == "POST" for request in server.requests)
    assert all("mcp-session-id" not in request.headers for request in server.requests)
    call = server.requests[-1]
    assert call.headers["mcp-name"] == header_value("поиск")
    assert rpc(call)["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"] == {}
    assert call.headers["authorization"] == "Bearer key"


async def test_the_handshake_mode_never_asks_a_modern_question() -> None:
    server = Server()
    async with client(server, protocol="legacy"):
        pass
    assert [message.get("method") for message in server.posts()][0] == "initialize"


@pytest.mark.parametrize(
    "answer",
    [
        refusal(400, -32022, "Unsupported protocol version", supported=["2027-01-01"]),
        httpx.Response(401),
        httpx.Response(307, headers={"location": "https://other.example.com/mcp"}),
    ],
)
async def test_a_modern_refusal_or_no_access_does_not_fall_back(answer: httpx.Response) -> None:
    sent: list[str] = []

    def server(request: httpx.Request) -> httpx.Response:
        sent.append(rpc(request)["method"])
        return answer

    with pytest.raises(McpError):
        await HttpMcpClient(URL, transport=httpx.MockTransport(server)).start()
    assert sent == ["server/discover"]


async def test_the_modern_mode_refuses_a_legacy_server() -> None:
    server = Server()
    with pytest.raises(McpError):
        await client(server, protocol="modern").start()
    assert [message.get("method") for message in server.posts()] == ["server/discover"]


async def test_a_request_for_input_is_an_error() -> None:
    def ask(request: httpx.Request) -> httpx.Response:
        result = {"resultType": "input_required", "inputRequests": {}, "requestState": "s"}
        return reply(rpc(request), result)

    async with client(Server(tools=ask, modern=True)) as mcp:
        with pytest.raises(McpError, match="input"):
            await mcp.list_tools()


def test_header_values_outside_plain_ascii_are_encoded() -> None:
    assert header_value("search_docs") == "search_docs"
    assert header_value("Hello, 世界") == "=?base64?SGVsbG8sIOS4lueVjA==?="
    assert header_value(" padded ") == "=?base64?IHBhZGRlZCA=?="
    assert header_value("=?base64?literal?=") == "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="


async def test_a_response_in_server_sent_events_with_a_request_of_the_server() -> None:
    def stream(request: httpx.Request) -> httpx.Response:
        call = rpc(request)["id"]
        result = json.dumps({"jsonrpc": "2.0", "id": call, "result": {"tools": [{"name": "a"}]}})
        ask = json.dumps({"jsonrpc": "2.0", "id": "srv-1", "method": "sampling/createMessage"})
        note = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
        return events(
            "id: 1\r\ndata:\r\n\r\n",
            ": a comment\n",
            f"data: {note}\n\n",
            f"event: message\ndata: {ask}\n\n",
            # The data of one event may come in several lines.
            "data: " + result.replace(", ", ",\ndata: ", 1) + "\n\n",
            "data: never read\n\n",
        )

    server = Server(tools=stream)
    async with client(server) as mcp:
        assert [tool.name for tool in await mcp.list_tools()] == ["a"]
    answers = [message for message in server.posts() if message.get("id") == "srv-1"]
    assert answers == [
        {"jsonrpc": "2.0", "id": "srv-1", "error": {"code": -32601, "message": "Method not found"}}
    ]


async def test_an_expired_session_starts_a_new_one_and_repeats_the_request() -> None:
    server = Server()
    calls: list[str] = []

    def tools(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["mcp-session-id"])
        if len(calls) == 1:
            server.session = "session-2"
            return httpx.Response(404)
        return reply(rpc(request), {"tools": []})

    server.tools = tools
    async with client(server) as mcp:
        assert await mcp.list_tools() == []
    assert calls == [SESSION, "session-2"]
    assert [message.get("method") for message in server.posts()].count("initialize") == 2


@pytest.mark.parametrize(
    ("response", "status", "text"),
    [
        (httpx.Response(401), 401, "access denied"),
        (httpx.Response(403), 403, "access denied"),
        (httpx.Response(302, headers={"location": "https://other.example.com/"}), 302, "redirect"),
        (
            httpx.Response(500, json={"jsonrpc": "2.0", "error": {"code": -1, "message": "boom"}}),
            500,
            "HTTP 500: boom",
        ),
    ],
)
async def test_refusals_of_the_server_are_errors_with_their_status(
    response: httpx.Response, status: int, text: str
) -> None:
    server = Server(tools=lambda request: response)
    async with client(server) as mcp:
        with pytest.raises(McpError) as caught:
            await mcp.list_tools()
    assert caught.value.status == status
    assert text in caught.value.message
    # A redirect is not followed.
    assert all(request.url.host == "mcp.example.com" for request in server.requests)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}),
        httpx.Response(200, content=b"not json", headers={"content-type": "application/json"}),
        httpx.Response(202),
        events("data: {}\n\n"),
    ],
)
async def test_a_broken_answer_is_a_protocol_error(response: httpx.Response) -> None:
    async with client(Server(tools=lambda request: response)) as mcp:
        with pytest.raises(McpError):
            await mcp.list_tools()


async def test_a_too_large_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import mcp_http

    monkeypatch.setattr(mcp_http, "MAX_MESSAGE_BYTES", 1000)
    big = {"tools": [{"name": "a", "description": "x" * 2000}]}
    async with client(Server(tools=lambda request: reply(rpc(request), big))) as mcp:
        with pytest.raises(McpError, match="too large"):
            await mcp.list_tools()
    streamed = "data: " + json.dumps({"jsonrpc": "2.0", "id": 3, "result": big}) + "\n\n"
    async with client(Server(tools=lambda request: events(streamed))) as mcp:
        with pytest.raises(McpError, match="too large"):
            await mcp.list_tools()


async def test_an_old_protocol_version_is_refused() -> None:
    def old(request: httpx.Request) -> httpx.Response:
        message = rpc(request)
        return reply(message, {"protocolVersion": "2024-11-05", "capabilities": {}})

    mcp = HttpMcpClient(URL, transport=httpx.MockTransport(old))
    with pytest.raises(McpError, match="unsupported protocol version"):
        await mcp.start()


async def test_an_unreachable_server_hides_the_address() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed for https://mcp.example.com/mcp?key=secret")

    mcp = HttpMcpClient(URL + "?key=secret", transport=httpx.MockTransport(down))
    with pytest.raises(McpError) as caught:
        await mcp.start()
    assert caught.value.message == "could not connect"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.com/mcp",
        "https://mcp.example.com:8443/mcp?key=1",
        "http://localhost:3000/mcp",
        "http://127.0.0.1:3000/mcp",
        "http://[::1]:3000/mcp",
    ],
)
def test_allowed_addresses(url: str) -> None:
    assert url_refusal(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "",
        "mcp.example.com/mcp",
        "ftp://mcp.example.com/mcp",
        "http://mcp.example.com/mcp",
        "http://192.168.1.5/mcp",
        "https://user:pass@mcp.example.com/mcp",
        "https://mcp.example.com/mcp#part",
        "https://169.254.169.254/latest",
        "https://[::ffff:169.254.169.254]/latest",
        "https://0.0.0.0/mcp",
        "https://mcp.example.com:99999/mcp",
        "https://mcp.example.com/m cp",
        "https://mcp.example.com/" + "a" * 2048,
    ],
)
def test_refused_addresses(url: str) -> None:
    assert url_refusal(url) is not None
    with pytest.raises(ValueError):
        HttpMcpClient(url)
