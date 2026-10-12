"""Single-user browser authentication: one-time login links and server-side sessions.

Secrets never reach the database in clear text: login codes and session ids are stored as
SHA-256 hashes. Comparisons with ``now`` happen in SQL so SQLite's naive datetimes and
PostgreSQL's aware ones behave the same way.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import Headers

from agent.config import Settings, normalize_origin
from agent.models import AuthBootstrapCode, AuthSession, utcnow

LAST_SEEN_RESOLUTION = timedelta(minutes=1)
LOGIN_FRAGMENT = "login"
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass(frozen=True, slots=True)
class IssuedSession:
    raw_id: str
    csrf_token: str
    expires_at: datetime


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def cookie_name(settings: Settings) -> str:
    # The __Host- prefix requires Secure, Path=/ and no Domain, pinning the cookie to the origin.
    return "__Host-agent_session" if settings.secure_cookies else "agent_session"


def host_allowed(settings: Settings, host_header: str | None) -> bool:
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith("["):
        host = host[: host.find("]") + 1] if "]" in host else host
    else:
        host = host.split(":", 1)[0]
    return host in {item.lower() for item in settings.allowed_hosts}


def origin_allowed(settings: Settings, origin: str) -> bool:
    try:
        return normalize_origin(origin) in settings.allowed_origins()
    except ValueError:
        return False


def token_valid(settings: Settings, supplied: str) -> bool:
    return bool(settings.api_token) and secrets.compare_digest(supplied, settings.api_token)


def login_url(settings: Settings, code: str) -> str:
    # The code travels in the fragment, so it never reaches proxies or access logs.
    return f"{settings.effective_public_origin()}/#{LOGIN_FRAGMENT}={code}"


async def issue_login_code(db: AsyncSession, settings: Settings) -> str:
    code = secrets.token_urlsafe(32)
    db.add(
        AuthBootstrapCode(
            code_hash=_digest(code),
            expires_at=utcnow() + timedelta(seconds=settings.auth_link_ttl_seconds),
        )
    )
    await db.commit()
    return code


async def consume_login_code(
    db: AsyncSession, settings: Settings, code: str, user_agent: str
) -> IssuedSession | None:
    now = utcnow()
    claimed = cast(
        CursorResult[Any],
        await db.execute(
            update(AuthBootstrapCode)
            .where(
                AuthBootstrapCode.code_hash == _digest(code),
                AuthBootstrapCode.consumed_at.is_(None),
                AuthBootstrapCode.expires_at > now,
            )
            .values(consumed_at=now)
        ),
    )
    if claimed.rowcount != 1:
        await db.rollback()
        return None

    raw_id = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires_at = now + timedelta(seconds=settings.auth_session_ttl_seconds)
    db.add(
        AuthSession(
            id_hash=_digest(raw_id),
            csrf_token=csrf_token,
            user_agent=user_agent[:512],
            created_at=now,
            expires_at=expires_at,
            last_seen=now,
        )
    )
    await db.commit()
    return IssuedSession(raw_id, csrf_token, expires_at)


async def authenticate(db: AsyncSession, raw_id: str | None) -> AuthSession | None:
    if not raw_id:
        return None
    now = utcnow()
    auth = (
        await db.execute(
            select(AuthSession).where(
                AuthSession.id_hash == _digest(raw_id),
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > now,
            )
        )
    ).scalar_one_or_none()
    if auth is not None and now - _aware(auth.last_seen) > LAST_SEEN_RESOLUTION:
        auth.last_seen = now
        await db.commit()
    return auth


async def revoke(db: AsyncSession, id_hash: str | None = None) -> int:
    """Revoke one session, or every active session when ``id_hash`` is None."""
    statement = update(AuthSession).where(AuthSession.revoked_at.is_(None))
    if id_hash is not None:
        statement = statement.where(AuthSession.id_hash == id_hash)
    result = cast(CursorResult[Any], await db.execute(statement.values(revoked_at=utcnow())))
    await db.commit()
    return int(result.rowcount)


async def has_active_session(db: AsyncSession) -> bool:
    found = await db.execute(
        select(AuthSession.id_hash)
        .where(AuthSession.revoked_at.is_(None), AuthSession.expires_at > utcnow())
        .limit(1)
    )
    return found.scalar_one_or_none() is not None


def session_expiry(auth: AuthSession) -> datetime:
    return _aware(auth.expires_at)


@dataclass(frozen=True, slots=True)
class BrowserCheck:
    """Outcome of validating a request's Host, Origin and credentials."""

    status: int
    detail: str = ""
    auth: AuthSession | None = None
    via_token: bool = False

    @property
    def ok(self) -> bool:
        return self.status == 200


async def check_request(
    db: AsyncSession,
    settings: Settings,
    headers: Headers,
    cookies: dict[str, str],
    method: str,
    *,
    require_origin: bool,
) -> BrowserCheck:
    """Shared policy for REST and WebSocket.

    A present Origin must be allowed even with a valid token. Browser mutations and
    WebSocket handshakes must carry an Origin; only token-authenticated clients (CLI,
    healthchecks) may omit it.
    """
    if not host_allowed(settings, headers.get("host")):
        return BrowserCheck(400, "Invalid host")
    origin = headers.get("origin")
    if origin is not None and not origin_allowed(settings, origin):
        return BrowserCheck(403, "Origin is not allowed")

    supplied_token = headers.get("x-agent-token")
    if supplied_token is not None:
        if token_valid(settings, supplied_token):
            return BrowserCheck(200, via_token=True)
        return BrowserCheck(401, "Unauthorized")

    auth = await authenticate(db, cookies.get(cookie_name(settings)))
    if auth is None:
        return BrowserCheck(401, "Login required")
    if origin is None and (require_origin or method in MUTATING_METHODS):
        return BrowserCheck(403, "Origin header is required")
    if method in MUTATING_METHODS and not secrets.compare_digest(
        headers.get("x-agent-csrf", ""), auth.csrf_token
    ):
        return BrowserCheck(403, "CSRF token mismatch")
    return BrowserCheck(200, auth=auth)
