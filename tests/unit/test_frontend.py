from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.testclient import TestClient

from agent.frontend import frontend_response


@pytest.fixture
def frontend_dist(tmp_path: Path) -> Path:
    directory = tmp_path / "dist"
    directory.mkdir()
    (directory / "index.html").write_text("<html>workspace</html>")
    return directory


APP_PATHS = ["", "login", "settings/models", "s/s1", "s/6c7125ba-a565-439c-895d-590c31514f99"]
UNKNOWN_PATHS = [
    "api/unknown",
    "assets/missing.js",
    "missing.css",
    "s/",
    "s/a/b",
    "settings",
    "settings/x",
]


@pytest.mark.parametrize("path", APP_PATHS)
def test_frontend_routes_serve_index(frontend_dist: Path, path: str) -> None:
    response = frontend_response(frontend_dist, path)
    assert isinstance(response, FileResponse)
    assert response.path == frontend_dist / "index.html"


@pytest.mark.parametrize("path", [*UNKNOWN_PATHS, "../secret"])
def test_unknown_or_unsafe_routes_do_not_serve_index(frontend_dist: Path, path: str) -> None:
    response = frontend_response(frontend_dist, path)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_existing_asset_is_served(frontend_dist: Path) -> None:
    asset = frontend_dist / "app.js"
    asset.write_text("export const ready = true;")
    response = frontend_response(frontend_dist, "app.js")
    assert isinstance(response, FileResponse)
    assert response.path == asset


@pytest.mark.parametrize("exists", [True, False])
def test_symlink_outside_frontend_is_rejected(frontend_dist: Path, exists: bool) -> None:
    outside = frontend_dist.parent / "outside"
    outside.mkdir()
    if exists:
        (outside / "s1").write_text("secret")
    (frontend_dist / "s").symlink_to(outside, target_is_directory=True)
    response = frontend_response(frontend_dist, "s/s1")
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_index_symlink_outside_frontend_is_rejected(frontend_dist: Path) -> None:
    outside = frontend_dist.parent / "secret.html"
    outside.write_text("secret")
    (frontend_dist / "index.html").unlink()
    (frontend_dist / "index.html").symlink_to(outside)
    response = frontend_response(frontend_dist, "s/s1")
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_missing_index_returns_not_found(frontend_dist: Path) -> None:
    (frontend_dist / "index.html").unlink()
    response = frontend_response(frontend_dist, "s/s1")
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


@pytest.mark.parametrize("path", ["app.js", "index.html"])
def test_symlink_loop_returns_not_found(frontend_dist: Path, path: str) -> None:
    target = frontend_dist / path
    target.unlink(missing_ok=True)
    target.symlink_to(target.name)
    response = frontend_response(frontend_dist, "s/s1" if path == "index.html" else path)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_direct_session_route_and_reload_return_html(frontend_dist: Path) -> None:
    app = FastAPI()

    @app.get("/{path:path}", response_model=None)
    async def serve_frontend(path: str) -> FileResponse | JSONResponse:
        return frontend_response(frontend_dist, path)

    with TestClient(app) as client:
        for _attempt in range(2):
            response = client.get("/s/s1")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert response.text == "<html>workspace</html>"
