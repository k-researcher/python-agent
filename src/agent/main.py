from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from agent.api import router, supervisor, websocket_endpoint
from agent.config import get_settings
from agent.database import init_database
from agent.events import broker

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.validate_runtime_security()
    await init_database()
    await broker.start(settings, listen=True)
    yield
    await supervisor.shutdown()
    await broker.stop()


app = FastAPI(title="Python Agent", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def api_token_auth(request: Request, call_next: RequestResponseEndpoint) -> Response:
    protected = request.url.path.startswith(("/api", "/docs", "/openapi.json", "/redoc"))
    if settings.api_token and protected:
        supplied = request.headers.get("X-Agent-Token", "")
        if not secrets.compare_digest(supplied, settings.api_token):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


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
    uvicorn.run("agent.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
