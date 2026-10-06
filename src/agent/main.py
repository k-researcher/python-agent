from __future__ import annotations

import sys
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from agent.api import router, supervisor, websocket_endpoint
from agent.auth import check_request, has_active_session, host_allowed, issue_login_code, login_url
from agent.auth_api import router as auth_router
from agent.config import get_settings
from agent.database import init_database, session_factory
from agent.events import broker
from agent.settings_api import router as settings_router

settings = get_settings()
PROTECTED_PREFIXES = ("/api", "/docs", "/openapi.json", "/redoc")
PUBLIC_PATHS = frozenset({"/api/health"})


async def announce_login_link() -> None:
    """Print a one-time login link when nobody is logged in yet (Jupyter-style)."""
    async with session_factory() as db:
        if await has_active_session(db):
            return
        url = login_url(settings, await issue_login_code(db, settings))
    minutes = settings.auth_link_ttl_seconds // 60
    print(
        f"\n  Python Agent: ссылка для входа (одноразовая, {minutes} мин):\n  {url}\n",
        flush=True,
    )
    if settings.auth_open_browser and settings.binds_loopback and sys.stdout.isatty():
        webbrowser.open(url)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.validate_runtime_security()
    await init_database()
    await broker.start(settings, listen=True)
    await announce_login_link()
    yield
    await supervisor.shutdown()
    await broker.stop()


app = FastAPI(title="Python Agent", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def browser_auth(request: Request, call_next: RequestResponseEndpoint) -> Response:
    # Exact Host matching defeats DNS rebinding against the local API.
    if not host_allowed(settings, request.headers.get("host")):
        return JSONResponse({"detail": "Invalid host"}, status_code=400)
    path = request.url.path
    if path in PUBLIC_PATHS or not path.startswith(PROTECTED_PREFIXES):
        return await call_next(request)
    async with session_factory() as db:
        check = await check_request(
            db, settings, request.headers, request.cookies, request.method, require_origin=False
        )
    if not check.ok:
        return JSONResponse({"detail": check.detail}, status_code=check.status)
    return await call_next(request)


app.include_router(auth_router)
app.include_router(settings_router)
app.include_router(router)
app.add_api_websocket_route("/ws", websocket_endpoint)


if settings.frontend_dist.is_dir():
    assets = settings.frontend_dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def frontend(path: str) -> FileResponse | JSONResponse:
        if path:
            try:
                frontend_root = settings.frontend_dist.resolve(strict=True)
                candidate = (frontend_root / path).resolve(strict=True)
                candidate.relative_to(frontend_root)
            except (OSError, ValueError):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            if candidate.is_file():
                return FileResponse(candidate)
        index = settings.frontend_dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse({"status": "frontend is not built"}, status_code=404)
else:
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            {
                "service": "python-agent",
                "status": "ok",
                "docs": "/docs",
                "frontend": "run npm install && npm run build in frontend/",
            }
        )


def run() -> None:
    uvicorn.run(
        "agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        proxy_headers=bool(settings.trusted_proxy_ips),
        forwarded_allow_ips=settings.trusted_proxy_ips or None,
    )


if __name__ == "__main__":
    run()
