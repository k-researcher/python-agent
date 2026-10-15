"""Async client for the agent HTTP API and WebSocket event stream.

The client authenticates with an API token sent in the ``X-Agent-Token`` header
on every request. The token never appears in exceptions, repr output or URLs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

MAX_RECONNECT_DELAY = 30.0
INITIAL_RECONNECT_DELAY = 1.0
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
TOKEN_HINT = (
    "No API token configured. Set the AGENT_API_TOKEN environment variable "
    "or pass a token to ClientConfig.from_env()."
)

WsResult = AsyncIterator[Any] | Awaitable[AsyncIterator[Any]]
WsConnect = Callable[[str], WsResult]
SleepFn = Callable[[float], Awaitable[Any]]


class ApiError(RuntimeError):
    """An API request failed with a non-2xx HTTP status or a transport error.

    ``status`` is the HTTP status code; ``detail`` is the server-provided message
    or a short description of the transport failure.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail

    def __str__(self) -> str:
        return self.detail

    def __repr__(self) -> str:
        return f"ApiError(status={self.status}, detail={self.detail!r})"


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Connection settings for :class:`AgentApi`."""

    base_url: str
    token: str

    def __init__(self, base_url: str, token: str) -> None:
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "token", token)

    @classmethod
    def from_env(cls, base_url: str | None = None, token: str | None = None) -> ClientConfig:
        if base_url is None:
            base_url = os.environ.get("AGENT_URL") or DEFAULT_BASE_URL
        if token is None:
            token = os.environ.get("AGENT_API_TOKEN", "")
        return cls(base_url, token)

    def __repr__(self) -> str:
        # Never leak the token into logs or error output.
        return f"ClientConfig(base_url={self.base_url!r}, token='***')"


def _status_from_exception(exc: BaseException) -> int | None:
    """Best-effort HTTP status for an exception a transport may raise."""
    if isinstance(exc, ApiError):
        return exc.status
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _http_status_detail(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and status in (401, 403):
            return "API server rejected the token (401 Unauthorized)"
    raw = getattr(exc, "body", None)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        if isinstance(data, dict):
            detail = data.get("detail")
            if isinstance(detail, str):
                return detail
        return raw
    message = str(exc)
    lowered = message.lower()
    if "401" in lowered:
        return "API server rejected the token (401 Unauthorized)"
    if "403" in lowered:
        return "API server rejected the token (403 Forbidden)"
    return message or "The connection was rejected"


def _ws_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"}:
        scheme = "ws" if parts.scheme == "wss" else "ws"
        parts = parts._replace(scheme=scheme)
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = parts.path.rstrip("/") + "/ws"
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


class AgentApi:
    """Thin async client for the agent REST API and WebSocket events."""

    _ws_connect: WsConnect
    _sleep: SleepFn

    def __init__(
        self, config: ClientConfig, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={"X-Agent-Token": config.token},
            transport=transport,
        )
        self._ws_connect = self._default_ws_connect
        self._sleep = self._default_sleep

    async def _default_sleep(self, delay: float) -> None:
        await asyncio.sleep(delay)

    async def _default_ws_connect(self, uri: str) -> AsyncIterator[Any]:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        connection = connect(
            uri,
            additional_headers={"X-Agent-Token": self._config.token},
            proxy=None,
        )
        try:
            async with connection as websocket:
                async for message in websocket:
                    yield message
        except InvalidStatus as exc:
            status = _status_from_exception(exc)
            if status in (401, 403):
                raise ApiError(status, _http_status_detail(exc)) from exc
            raise
        except ConnectionClosed:
            # Drop to the outer reconnect loop.
            return

    def _redact(self, error: ApiError) -> ApiError:
        """Return the error with every copy of the token replaced by ``***``."""
        token = self._config.token
        if token and token in error.detail:
            return ApiError(error.status, error.detail.replace(token, "***"))
        return error

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release its resources."""
        await self._client.aclose()

    # -- HTTP helpers -------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return await self._send(method, path, json_body)
        except ApiError as exc:
            raise self._redact(exc) from exc.__cause__

    async def _send(self, method: str, path: str, json_body: dict[str, Any] | None) -> Any:
        if not self._config.token:
            raise ApiError(401, TOKEN_HINT)
        try:
            response = await self._client.request(
                method, path, json=json_body if json_body is not None else None
            )
        except httpx.HTTPError as exc:
            raise ApiError(0, f"Request failed: {exc}") from exc
        if response.is_error:
            raise await self._error_from(response)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(response.status_code, f"Invalid JSON in response: {exc}") from exc

    async def _error_from(self, response: httpx.Response) -> ApiError:
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("detail"), str):
            detail = data["detail"]
        else:
            detail = response.text or response.reason_phrase or "Request failed"
        return ApiError(response.status_code, detail)

    # -- Projects -----------------------------------------------------------

    async def projects(self) -> list[dict[str, Any]]:
        """List every registered project."""
        data = await self._request("GET", "/api/projects")
        if not isinstance(data, list):
            raise ApiError(0, "Projects endpoint returned an unexpected payload")
        return data

    async def ensure_project(self, root_path: str, name: str | None = None) -> dict[str, Any]:
        """Return the project registered for ``root_path`` or create it."""
        if not root_path or not root_path.strip():
            raise ApiError(0, "Project root path must not be empty")
        absolute = Path(root_path).expanduser().resolve()
        try:
            existing = await self.projects()
        except ApiError:
            raise
        for project in existing:
            if not isinstance(project, dict):
                continue
            registered = project.get("root_path")
            if isinstance(registered, str) and Path(registered).resolve() == absolute:
                return project
        project_name = (name or "").strip() or absolute.name or "project"
        payload = {"root_path": str(absolute), "name": project_name}
        data = await self._request("POST", "/api/projects", json_body=payload)
        if not isinstance(data, dict):
            raise ApiError(0, "Create project endpoint returned an unexpected payload")
        return data

    # -- Sessions -----------------------------------------------------------

    async def sessions(self) -> list[dict[str, Any]]:
        """List every active session."""
        data = await self._request("GET", "/api/sessions")
        if not isinstance(data, list):
            raise ApiError(0, "Sessions endpoint returned an unexpected payload")
        return data

    async def create_session(
        self,
        project_id: str,
        prompt: str,
        *,
        mode: str = "dev",
        llm_profile: str | None = None,
        reasoning_effort: str | None = None,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        """Create a new agent session for ``project_id``."""
        payload: dict[str, Any] = {
            "project_id": project_id,
            "prompt": prompt,
            "mode": mode,
            "auto_start": auto_start,
        }
        if llm_profile is not None:
            payload["llm_profile"] = llm_profile
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        data = await self._request("POST", "/api/sessions", json_body=payload)
        if not isinstance(data, dict):
            raise ApiError(0, "Create session endpoint returned an unexpected payload")
        return data

    async def session(self, session_id: str) -> dict[str, Any]:
        """Return one session by id."""
        data = await self._request("GET", f"/api/sessions/{session_id}")
        if not isinstance(data, dict):
            raise ApiError(0, "Session endpoint returned an unexpected payload")
        return data

    async def messages(self, session_id: str) -> list[dict[str, Any]]:
        """List every message in a session."""
        data = await self._request("GET", f"/api/sessions/{session_id}/messages")
        if not isinstance(data, list):
            raise ApiError(0, "Messages endpoint returned an unexpected payload")
        return data

    async def send_message(
        self,
        session_id: str,
        content: str,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Add a user message to a session and resume the agent."""
        payload: dict[str, Any] = {"content": content}
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        data = await self._request(
            "POST", f"/api/sessions/{session_id}/messages", json_body=payload
        )
        if not isinstance(data, dict):
            raise ApiError(0, "Send message endpoint returned an unexpected payload")
        return data

    async def approvals(self, session_id: str) -> list[dict[str, Any]]:
        """List pending approvals for a session."""
        data = await self._request("GET", f"/api/sessions/{session_id}/approvals")
        if not isinstance(data, list):
            raise ApiError(0, "Approvals endpoint returned an unexpected payload")
        return data

    async def resolve_approval(
        self, approval_id: str, decision: str, comment: str | None = None
    ) -> None:
        """Approve or reject one pending approval."""
        payload: dict[str, Any] = {"decision": decision}
        if comment is not None:
            payload["comment"] = comment
        await self._request("POST", f"/api/approvals/{approval_id}", json_body=payload)

    async def stop(self, session_id: str) -> None:
        """Stop a running session."""
        await self._request("POST", f"/api/sessions/{session_id}/stop")

    async def start(self, session_id: str) -> None:
        """Start or resume a session."""
        await self._request("POST", f"/api/sessions/{session_id}/start")

    # -- WebSocket events ---------------------------------------------------

    async def events(self, session_id: str | None = None) -> AsyncGenerator[dict[str, Any], None]:
        """Yield API events, reconnecting on transient failures.

        Events are emitted as JSON objects. When ``session_id`` is given, only
        events for that session are yielded. Reconnection uses the delays
        1, 2, 4 ... 30 seconds. A 401/403 response stops the stream with
        :class:`ApiError`.
        """
        if not self._config.token:
            raise ApiError(401, TOKEN_HINT)
        delay = float(INITIAL_RECONNECT_DELAY)
        uri = _ws_url(self._config.base_url)
        while True:
            try:
                result: WsResult = self._ws_connect(uri)
                if inspect.isawaitable(result):
                    iterator = await result
                else:
                    iterator = result
                async for raw in iterator:
                    event = self._decode_event(raw)
                    if event is None:
                        continue
                    if session_id is not None and event.get("session_id") != session_id:
                        continue
                    yield event
                    delay = float(INITIAL_RECONNECT_DELAY)
            except ApiError as exc:
                raise self._redact(exc) from exc.__cause__
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                status = _status_from_exception(exc)
                if status in (401, 403):
                    raise self._redact(ApiError(status, _http_status_detail(exc))) from exc
                await self._sleep(delay)
                delay = min(delay * 2.0, MAX_RECONNECT_DELAY)

    @staticmethod
    def _decode_event(raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (str, bytes, bytearray)):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None
        return None
