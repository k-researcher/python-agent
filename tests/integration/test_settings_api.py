"""Verify settings mutations behind the application's browser security middleware."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from agent import settings_api
from agent.config import get_settings
from agent.database import session_factory
from agent.main import app
from agent.model_overrides import (
    LLMModelOverride,
    LLMProviderOverride,
    LLMRoutingOverride,
    load_effective_registry,
)
from agent.models import Event, OutboundAudit
from tests.integration.helpers import authed_client

PREFIX = "/api/settings/models"
FILE_KEY = "file-secret-abcd"
UI_KEY = "ui-private-secret-1234"


async def clean_overrides() -> None:
    async with session_factory() as db:
        for table in (LLMRoutingOverride, LLMModelOverride, LLMProviderOverride):
            await db.execute(delete(table))
        await db.execute(delete(Event).where(Event.type == "settings.models.changed"))
        await db.execute(delete(OutboundAudit).where(OutboundAudit.operation == "list_models"))
        await db.commit()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    path = tmp_path / "models.yaml"
    path.write_text(
        """providers:
  cloud:
    kind: openai
    base_url: https://example.com/v1
    api_key_env: SETTINGS_TEST_KEY
  idle:
    kind: openai
    base_url: http://localhost:1234/v1
models:
  fast:
    provider: cloud
    model: remote-fast
    context_window: 16000
    max_tokens: 2000
    reasoning:
      supported: true
      allowed_efforts: [low, high]
      default_effort: low
      wire_parameter: reasoning_effort
  spare:
    provider: cloud
    model: remote-spare
    context_window: 8000
    max_tokens: 1000
routing:
  default: fast
  fallback: []
  roles: {review: fast}
"""
    )
    monkeypatch.setenv("AGENT_MODELS_FILE", str(path))
    monkeypatch.setenv("SETTINGS_TEST_KEY", FILE_KEY)
    monkeypatch.setattr(get_settings(), "models_file", path, raising=False)
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    with authed_client() as authenticated:
        assert authenticated.portal is not None
        authenticated.portal.call(clean_overrides)
        yield authenticated
        authenticated.portal.call(clean_overrides)


def provider(view: dict[str, Any], provider_id: str = "cloud") -> dict[str, Any]:
    return next(item for item in view["providers"] if item["id"] == provider_id)


def model(view: dict[str, Any], model_id: str = "fast") -> dict[str, Any]:
    return next(item for item in view["models"] if item["id"] == model_id)


def test_requires_login() -> None:
    with TestClient(app) as anonymous:
        assert anonymous.get(PREFIX).status_code == 401
        assert anonymous.put(f"{PREFIX}/routing", json={"default": "fast"}).status_code == 401


def test_get_and_csrf_protection(client: TestClient) -> None:
    response = client.get(PREFIX)
    assert response.status_code == 200
    body = response.json()
    assert provider(body)["api_key_set"] is True
    assert provider(body)["api_key_hint"] == "…abcd"
    assert provider(body)["sources"]["api_key"] == "env"
    assert provider(body)["origin"] == "file"
    assert FILE_KEY not in response.text
    denied = client.put(
        f"{PREFIX}/models/fast",
        json={"max_tokens": 3000},
        headers={"X-Agent-CSRF": "invalid"},
    )
    assert denied.status_code == 403
    assert model(client.get(PREFIX).json())["max_tokens"] == 2000


def test_reasoning_field_merge_and_reset(client: TestClient) -> None:
    changed = client.put(f"{PREFIX}/models/fast", json={"reasoning": {"supported": False}})
    assert changed.status_code == 200, changed.text
    entry = model(changed.json())
    assert entry["reasoning"] == {
        "supported": False,
        "allowed_efforts": [],
        "default_effort": None,
        "wire_parameter": None,
        "history": "none",
    }
    assert entry["max_tokens"] == 2000
    assert entry["context_window"] == 16000
    assert entry["sources"]["reasoning"] == "ui"
    assert entry["sources"]["context_window"] == "file"
    reset = client.delete(f"{PREFIX}/models/fast")
    assert reset.status_code == 200, reset.text
    assert model(reset.json())["reasoning"]["default_effort"] == "low"
    assert model(reset.json())["origin"] == "file"

    async def check_reset() -> None:
        async with session_factory() as db:
            assert await db.get(LLMModelOverride, "fast") is None

    assert client.portal is not None
    client.portal.call(check_reset)


def test_new_ui_model_and_provider_then_delete(client: TestClient) -> None:
    created_provider = client.put(
        f"{PREFIX}/providers/local", json={"kind": "openai", "base_url": "http://localhost:123/v1"}
    )
    assert created_provider.status_code == 200, created_provider.text
    created = client.put(
        f"{PREFIX}/models/custom",
        json={"provider": "local", "model": "custom", "context_window": 8000, "max_tokens": 1000},
    )
    assert created.status_code == 200, created.text
    assert model(created.json(), "custom")["origin"] == "ui"
    assert set(model(created.json(), "custom")["sources"].values()) == {"ui"}
    assert client.delete(f"{PREFIX}/providers/local").status_code == 422
    deleted = client.delete(f"{PREFIX}/models/custom")
    assert deleted.status_code == 200
    assert not any(item["id"] == "custom" for item in deleted.json()["models"])
    assert client.delete(f"{PREFIX}/providers/local").status_code == 200
    assert client.delete(f"{PREFIX}/models/absent").status_code == 404


def test_disabled_and_reset(client: TestClient) -> None:
    disabled = client.delete(f"{PREFIX}/models/spare?disable=true")
    assert disabled.status_code == 200, disabled.text
    assert model(disabled.json(), "spare")["disabled"] is True
    assert client.delete(f"{PREFIX}/providers/idle?disable=true").status_code == 200

    async def check_runtime() -> None:
        async with session_factory() as db:
            registry = await load_effective_registry(db, get_settings())
            assert "spare" not in registry.models
            row = await db.get(LLMProviderOverride, "idle")
            assert row is not None and row.disabled

    assert client.portal is not None
    client.portal.call(check_runtime)
    reset = client.delete(f"{PREFIX}/models/spare")
    assert reset.status_code == 200
    assert model(reset.json(), "spare")["disabled"] is False
    assert client.delete(f"{PREFIX}/providers/idle").status_code == 200
    assert client.delete(f"{PREFIX}/models/fast?disable=true").status_code == 422


def test_encryption_hints_env_switch_and_change_events(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = client.get(PREFIX).json()["checksum"]
    changed = client.put(f"{PREFIX}/providers/cloud", json={"api_key": UI_KEY})
    assert changed.status_code == 200, changed.text
    assert provider(changed.json())["api_key_hint"] == "…1234"
    assert provider(changed.json())["sources"]["api_key"] == "ui"
    assert changed.json()["checksum"] != before
    assert UI_KEY not in changed.text
    assert UI_KEY not in client.get(PREFIX).text

    async def check_encryption() -> None:
        async with session_factory() as db:
            row = await db.get(LLMProviderOverride, "cloud")
            assert row is not None and row.api_key_ciphertext
            assert UI_KEY not in row.api_key_ciphertext
            assert "api_key" not in row.fields
            registry = await load_effective_registry(db, get_settings())
            assert registry.get("fast").api_key == UI_KEY
            events = (
                await db.scalars(select(Event).where(Event.type == "settings.models.changed"))
            ).all()
            assert len(events) == 1
            assert events[0].payload == {"checksum": changed.json()["checksum"]}
            assert UI_KEY not in json.dumps(events[0].payload)

    assert client.portal is not None
    client.portal.call(check_encryption)
    monkeypatch.setenv("SETTINGS_OTHER_KEY", "other-key-wxyz")
    switched = client.put(f"{PREFIX}/providers/cloud", json={"api_key_env": "SETTINGS_OTHER_KEY"})
    assert switched.status_code == 200, switched.text
    assert provider(switched.json())["api_key_hint"] == "…wxyz"
    assert provider(switched.json())["sources"]["api_key"] == "env"
    assert client.delete(f"{PREFIX}/providers/cloud").status_code == 200
    assert provider(client.get(PREFIX).json())["api_key_hint"] == "…abcd"


def test_invalid_changes_are_422_without_writes_or_secret_echo(client: TestClient) -> None:
    assert client.put(f"{PREFIX}/models/fast", json={"max_tokens": 3000}).status_code == 200
    before = client.get(PREFIX).json()["checksum"]
    invalid = client.put(f"{PREFIX}/models/fast", json={"max_tokens": 16000})
    assert invalid.status_code == 422, invalid.text
    assert client.get(PREFIX).json()["checksum"] == before
    assert model(client.get(PREFIX).json())["max_tokens"] == 3000
    assert client.put(f"{PREFIX}/models/new", json={"model": "incomplete"}).status_code == 422
    assert client.put(f"{PREFIX}/models/fast", json={"provider": "absent"}).status_code == 422
    bad_reasoning = client.put(
        f"{PREFIX}/models/fast", json={"reasoning": {"default_effort": "high"}}
    )
    assert bad_reasoning.status_code == 422
    for fields in (
        {"api_key": UI_KEY, "base_url": "invalid"},
        {"api_key": UI_KEY, "api_key_env": "SETTINGS_TEST_KEY"},
        {"api_key": UI_KEY, "kind": "invalid"},
        {"api_key": UI_KEY, "unexpected": UI_KEY},
    ):
        response = client.put(f"{PREFIX}/providers/cloud", json=fields)
        assert response.status_code == 422, response.text
        assert UI_KEY not in response.text

    async def check_no_write() -> None:
        async with session_factory() as db:
            assert await db.get(LLMModelOverride, "new") is None
            assert await db.get(LLMProviderOverride, "cloud") is None
            events = (
                await db.scalars(select(Event).where(Event.type == "settings.models.changed"))
            ).all()
            assert len(events) == 1

    assert client.portal is not None
    client.portal.call(check_no_write)


def test_routing_override(client: TestClient) -> None:
    changed = client.put(
        f"{PREFIX}/routing",
        json={"default": "spare", "fallback": ["fast"], "roles": {"arch": "fast"}},
    )
    assert changed.status_code == 200, changed.text
    routing = changed.json()["routing"]
    assert routing["default"] == "spare"
    assert routing["fallback"] == ["fast"]
    assert routing["roles"] == {"arch": "fast"}
    assert set(routing["sources"].values()) == {"ui"}
    assert client.put(f"{PREFIX}/routing", json={"default": "absent"}).status_code == 422
    assert client.get(PREFIX).json()["routing"] == routing


def test_connection_uses_key_and_writes_empty_payload_audit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.put(f"{PREFIX}/providers/cloud", json={"api_key": UI_KEY}).status_code == 200

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/v1/models"
        assert request.headers["Authorization"] == f"Bearer {UI_KEY}"
        assert request.extensions["timeout"]["read"] == 10
        return httpx.Response(200, json={"data": [{"id": "model-one"}, {"id": UI_KEY}]})

    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond), timeout=10, follow_redirects=False
        ),
    )
    response = client.post(f"{PREFIX}/providers/cloud/test")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": 200, "models": ["model-one"]}
    assert UI_KEY not in response.text

    async def check_audit() -> None:
        async with session_factory() as db:
            row = (
                await db.scalars(
                    select(OutboundAudit).where(OutboundAudit.operation == "list_models")
                )
            ).one()
            assert row.category == "llm"
            assert row.payload_bytes == 0
            assert row.payload_sha256 == hashlib.sha256(b"").hexdigest()
            assert row.status == "completed"
            assert row.detail == "200"
            assert UI_KEY not in str(row.__dict__)

    assert client.portal is not None
    client.portal.call(check_audit)


@pytest.mark.parametrize("failure", ["timeout", "redirect", "malformed", "http_error"])
def test_connection_failures_are_safe_and_audited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if failure == "timeout":
            raise httpx.ReadTimeout(FILE_KEY, request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "https://redirect.example/models"})
        if failure == "malformed":
            return httpx.Response(200, text=FILE_KEY)
        return httpx.Response(401, text=FILE_KEY)

    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond), timeout=10, follow_redirects=False
        ),
    )
    response = client.post(f"{PREFIX}/providers/cloud/test")
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["models"] == []
    assert len(requests) == 1
    assert FILE_KEY not in response.text

    async def check_audit() -> None:
        async with session_factory() as db:
            row = (
                await db.scalars(
                    select(OutboundAudit).where(OutboundAudit.operation == "list_models")
                )
            ).one()
            assert row.status == "error"
            assert FILE_KEY not in str(row.__dict__)

    assert client.portal is not None
    client.portal.call(check_audit)


def test_legacy_base_when_yaml_is_absent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = get_settings().data_dir / "missing-models.yaml"
    # An explicit path that does not exist is an error, not a silent switch to legacy.
    monkeypatch.setattr(get_settings(), "models_file", str(missing), raising=False)
    assert client.get(PREFIX).status_code == 422
    # An empty value selects the legacy AGENT_LLM_* settings explicitly.
    monkeypatch.setattr(get_settings(), "models_file", "", raising=False)
    view = client.get(PREFIX)
    assert view.status_code == 200, view.text
    assert view.json()["models"][0]["id"] == "default"
    assert view.json()["models"][0]["origin"] == "legacy"
    changed = client.put(f"{PREFIX}/models/default", json={"max_tokens": 4000})
    assert changed.status_code == 200
    assert model(changed.json(), "default")["sources"]["model"] == "legacy"
    assert model(changed.json(), "default")["sources"]["max_tokens"] == "ui"


def test_models_in_use_by_sessions_cannot_be_removed(client: TestClient, tmp_path: Path) -> None:
    project = client.post(
        "/api/projects", json={"name": "In use", "root_path": str(tmp_path)}
    ).json()
    created = client.post(
        "/api/sessions",
        json={
            "project_id": project["id"],
            "prompt": "Use the spare model",
            "llm_profile": "spare",
            "auto_start": False,
        },
    )
    assert created.status_code == 201, created.text

    assert client.delete(f"{PREFIX}/models/spare?disable=true").status_code == 409
    assert client.put(f"{PREFIX}/models/spare", json={"disabled": True}).status_code == 409

    session_id = created.json()["id"]
    assert client.post(f"/api/sessions/{session_id}/stop").status_code == 200
    assert client.post(f"/api/sessions/{session_id}/archive").status_code == 200
    assert client.delete(f"{PREFIX}/models/spare?disable=true").status_code == 200
