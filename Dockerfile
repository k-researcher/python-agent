FROM node:22-alpine AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY prompts/ ./prompts/
COPY alembic.ini ./
COPY --from=frontend-build /build/frontend/dist ./frontend/dist/

RUN uv sync --frozen --no-dev
RUN groupadd --gid 10001 agent && \
    useradd --uid 10001 --gid agent --create-home agent && \
    mkdir -p /app/data && \
    chown agent:agent /app/data

EXPOSE 8080
USER agent
CMD [".venv/bin/agent"]
