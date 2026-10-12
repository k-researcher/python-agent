from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from agent.api import settings
from agent.auth import issue_login_code
from agent.database import session_factory
from agent.main import app

ORIGIN = "http://testserver"


def login(client: TestClient) -> str:
    """Log the client in through a one-time code and return the CSRF token."""

    async def issue() -> str:
        async with session_factory() as db:
            return await issue_login_code(db, settings)

    code = client.portal.call(issue)  # type: ignore[union-attr]
    response = client.post("/auth/login", json={"code": code}, headers={"Origin": ORIGIN})
    assert response.status_code == 200, response.text
    csrf = str(response.json()["csrf_token"])
    client.headers.update({"Origin": ORIGIN, "X-Agent-CSRF": csrf})
    return csrf


@contextmanager
def authed_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        login(client)
        yield client
