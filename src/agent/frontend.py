from __future__ import annotations

import re
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse

SESSION_ROUTE = re.compile(r"s/[A-Za-z0-9_-]+\Z")
# Client routes without a parameter; a direct link or a reload of them gets the app page.
APP_ROUTES = frozenset({"login", "settings/models"})


def frontend_response(directory: Path, path: str) -> FileResponse | JSONResponse:
    try:
        root = directory.resolve(strict=True)
        candidate = (root / path).resolve()
        candidate.relative_to(root)
        if path and candidate.is_file():
            return FileResponse(candidate)
        if path and path not in APP_ROUTES and SESSION_ROUTE.fullmatch(path) is None:
            return JSONResponse({"detail": "Not found"}, status_code=404)
        index = (root / "index.html").resolve(strict=True)
        index.relative_to(root)
        if index.is_file():
            return FileResponse(index)
    except (OSError, ValueError, RuntimeError):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return JSONResponse({"status": "frontend is not built"}, status_code=404)
