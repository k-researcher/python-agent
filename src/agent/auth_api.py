from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.auth import (
    authenticate,
    check_request,
    consume_login_code,
    cookie_name,
    host_allowed,
    origin_allowed,
    revoke,
    session_expiry,
)
from agent.config import get_settings
from agent.database import get_db

router = APIRouter(prefix="/auth")
settings = get_settings()
NO_STORE = {"Cache-Control": "no-store"}


class LoginRequest(BaseModel):
    code: str = Field(min_length=16, max_length=256)


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=NO_STORE)


@router.get("/session")
async def current_session(request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    if not host_allowed(settings, request.headers.get("host")):
        return _error(400, "Invalid host")
    auth = await authenticate(db, request.cookies.get(cookie_name(settings)))
    body: dict[str, Any] = {"authenticated": auth is not None}
    if auth is not None:
        body["expires_at"] = session_expiry(auth).isoformat()
        body["csrf_token"] = auth.csrf_token
    return JSONResponse(body, headers=NO_STORE)


@router.post("/login")
async def login(
    payload: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)
) -> Response:
    # No session or CSRF token exists yet, so Host, Origin and JSON are the guards here.
    if not host_allowed(settings, request.headers.get("host")):
        return _error(400, "Invalid host")
    origin = request.headers.get("origin")
    if origin is None or not origin_allowed(settings, origin):
        return _error(403, "Origin is not allowed")
    if not request.headers.get("content-type", "").startswith("application/json"):
        return _error(415, "JSON body is required")

    issued = await consume_login_code(
        db, settings, payload.code, request.headers.get("user-agent", "")
    )
    if issued is None:
        return _error(401, "Login link is invalid, used or expired")

    response = JSONResponse(
        {
            "authenticated": True,
            "expires_at": issued.expires_at.isoformat(),
            "csrf_token": issued.csrf_token,
        },
        headers=NO_STORE,
    )
    response.set_cookie(
        cookie_name(settings),
        issued.raw_id,
        max_age=settings.auth_session_ttl_seconds,
        path="/",
        secure=settings.secure_cookies,
        httponly=True,
        samesite="strict",
    )
    return response


async def _require_session(request: Request, db: AsyncSession) -> JSONResponse | str:
    check = await check_request(
        db, settings, request.headers, request.cookies, request.method, require_origin=True
    )
    if not check.ok or check.auth is None:
        return _error(check.status if not check.ok else 401, check.detail or "Login required")
    return check.auth.id_hash


@router.post("/logout")
async def logout(request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    outcome = await _require_session(request, db)
    if isinstance(outcome, JSONResponse):
        return outcome
    await revoke(db, outcome)
    response = Response(status_code=204, headers=NO_STORE)
    response.delete_cookie(
        cookie_name(settings),
        path="/",
        secure=settings.secure_cookies,
        httponly=True,
        samesite="strict",
    )
    return response


@router.post("/revoke-all")
async def revoke_all(request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    outcome = await _require_session(request, db)
    if isinstance(outcome, JSONResponse):
        return outcome
    revoked = await revoke(db)
    response = JSONResponse({"revoked": revoked}, headers=NO_STORE)
    response.delete_cookie(cookie_name(settings), path="/", secure=settings.secure_cookies)
    return response
