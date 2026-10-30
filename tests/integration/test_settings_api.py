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

from agent import model_health, settings_api
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
        "tags": None,
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


def test_reasoning_tags_can_be_set_and_are_validated(client: TestClient) -> None:
    tags = {"start": "<|reasoning>", "end": "<reasoning|>"}
    changed = client.put(
        f"{PREFIX}/models/fast", json={"reasoning": {"supported": False, "tags": tags}}
    )
    assert changed.status_code == 200, changed.text
    assert model(changed.json())["reasoning"]["tags"] == tags
    empty = client.put(
        f"{PREFIX}/models/fast",
        json={"reasoning": {"supported": False, "tags": {"start": "", "end": "x"}}},
    )
    assert empty.status_code == 422


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
    assert {routing["sources"][key] for key in ("default", "fallback", "roles")} == {"ui"}
    assert routing["sources"]["light"] == "file"
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
    assert response.json() == {
        "ok": True,
        "status": 200,
        "models": ["model-one"],
        "details": [
            {
                "id": "model-one",
                "context_window": None,
                "max_output_tokens": None,
                "kind": None,
                "state": None,
            }
        ],
    }
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


def _mock_connection(monkeypatch: pytest.MonkeyPatch, respond: Any) -> None:
    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond), timeout=10, follow_redirects=False
        ),
    )


def test_connection_reports_context_window_from_the_listing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "wide", "context_length": 131072},
                    {"id": "plain"},
                    {"id": "broken", "context_length": "large"},
                ]
            },
        )

    _mock_connection(monkeypatch, respond)
    details = client.post(f"{PREFIX}/providers/cloud/test").json()["details"]
    assert details == [
        {
            "id": "wide",
            "context_window": 131072,
            "max_output_tokens": None,
            "kind": None,
            "state": None,
        },
        {
            "id": "plain",
            "context_window": None,
            "max_output_tokens": None,
            "kind": None,
            "state": None,
        },
        {
            "id": "broken",
            "context_window": None,
            "max_output_tokens": None,
            "kind": None,
            "state": None,
        },
    ]
    # A remote provider gets only the standard listing request.
    assert requests == ["https://example.com/v1/models"]


def test_connection_reads_gateway_limits_and_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A gateway listing gives the input and output limits and the mode of each model.
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "chat",
                        "mode": "chat",
                        "max_input_tokens": 262144,
                        "max_output_tokens": 65536,
                    },
                    {"id": "vectors", "mode": "embedding", "max_input_tokens": 8192},
                    {"id": "bare"},
                ]
            },
        )

    _mock_connection(monkeypatch, respond)
    details = client.post(f"{PREFIX}/providers/cloud/test").json()["details"]
    assert details == [
        {
            "id": "chat",
            "context_window": 262144,
            "max_output_tokens": 65536,
            "kind": "chat",
            "state": None,
        },
        {
            "id": "vectors",
            "context_window": 8192,
            "max_output_tokens": None,
            "kind": "embedding",
            "state": None,
        },
        {
            "id": "bare",
            "context_window": None,
            "max_output_tokens": None,
            "kind": None,
            "state": None,
        },
    ]


def test_connection_reads_local_server_model_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "chat-model"}, {"id": "embedder"}]})
        assert "Authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "chat-model",
                        "type": "llm",
                        "max_context_length": 262144,
                        "state": "loaded",
                    },
                    {"id": "embedder", "type": "embeddings", "max_context_length": 2048},
                    {"id": "not-listed", "type": "llm", "max_context_length": 4096},
                ]
            },
        )

    _mock_connection(monkeypatch, respond)
    result = client.post(f"{PREFIX}/providers/idle/test").json()
    assert result["ok"] is True
    assert result["models"] == ["chat-model", "embedder"]
    assert result["details"] == [
        {
            "id": "chat-model",
            "context_window": 262144,
            "max_output_tokens": None,
            "kind": "llm",
            "state": "loaded",
        },
        {
            "id": "embedder",
            "context_window": 2048,
            "max_output_tokens": None,
            "kind": "embeddings",
            "state": None,
        },
    ]
    assert requests == ["http://localhost:1234/v1/models", "http://localhost:1234/api/v0/models"]

    async def check_audit() -> None:
        async with session_factory() as db:
            rows = (
                await db.scalars(
                    select(OutboundAudit)
                    .where(OutboundAudit.category == "llm")
                    .order_by(OutboundAudit.created_at)
                )
            ).all()
            assert [(row.operation, row.status) for row in rows] == [
                ("list_models", "completed"),
                ("list_model_details", "completed"),
            ]

    assert client.portal is not None
    client.portal.call(check_audit)


def test_connection_keeps_the_listing_when_local_details_fail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "chat-model"}]})
        return httpx.Response(404, text="Not found")

    _mock_connection(monkeypatch, respond)
    result = client.post(f"{PREFIX}/providers/idle/test").json()
    assert result["ok"] is True
    assert result["details"] == [
        {
            "id": "chat-model",
            "context_window": None,
            "max_output_tokens": None,
            "kind": None,
            "state": None,
        }
    ]


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


def test_model_id_with_a_slash_can_be_added_changed_and_removed(client: TestClient) -> None:
    # Local servers name models like "publisher/model"; the ID keeps the slash.
    model_id = "publisher/chat-model"
    body = {"provider": "idle", "model": model_id, "context_window": 32768, "max_tokens": 4096}
    response = client.put(f"{PREFIX}/models/{model_id}", json=body)
    assert response.status_code == 200
    assert model(response.json(), model_id)["context_window"] == 32768
    response = client.put(f"{PREFIX}/models/{model_id}", json={"max_tokens": 2048})
    assert model(response.json(), model_id)["max_tokens"] == 2048
    response = client.delete(f"{PREFIX}/models/{model_id}")
    assert response.status_code == 200
    assert all(item["id"] != model_id for item in response.json()["models"])


def test_probe_sends_one_token_request_and_reports_the_answer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {FILE_KEY}"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "o"}}]})

    _mock_connection(monkeypatch, respond)
    result = client.post(f"{PREFIX}/providers/cloud/probe", json={"model": "remote-fast"}).json()
    assert result["ok"] is True
    assert result["status"] == 200
    assert isinstance(result["seconds"], float)
    assert sent == [
        {"model": "remote-fast", "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}
    ]

    async def check_audit() -> None:
        async with session_factory() as db:
            row = (
                await db.scalars(
                    select(OutboundAudit).where(OutboundAudit.operation == "probe_model")
                )
            ).one()
            assert row.status == "completed"
            assert row.payload_bytes > 0
            assert FILE_KEY not in str(row.__dict__)

    assert client.portal is not None
    client.portal.call(check_audit)


@pytest.mark.parametrize("failure", ["http_error", "timeout"])
def test_probe_reports_a_dead_model_without_its_error_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        # A gateway error may name internal hosts; it stays out of the answer.
        return httpx.Response(500, json={"error": {"message": "Cannot connect to host 10.0.0.5"}})

    _mock_connection(monkeypatch, respond)
    result = client.post(f"{PREFIX}/providers/cloud/probe", json={"model": "dead"}).json()
    assert result["ok"] is False
    assert result["status"] == (500 if failure == "http_error" else None)
    assert "10.0.0.5" not in json.dumps(result)


def test_probe_rejects_unknown_provider_and_empty_model(client: TestClient) -> None:
    assert client.post(f"{PREFIX}/providers/absent/probe", json={"model": "m"}).status_code == 404
    assert client.post(f"{PREFIX}/providers/cloud/probe", json={"model": ""}).status_code == 422


def test_local_server_scan_finds_running_servers_and_marks_connected_ones(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"port": 1235}))
    monkeypatch.setattr(settings_api, "LOCAL_SERVER_PORTS", (1234, 11434))
    monkeypatch.setattr(settings_api, "local_port_files", lambda: [config])

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        port, path = request.url.port, request.url.path
        if port == 1235 and path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
        if port == 1235 and path == "/api/v0/models":
            return httpx.Response(200, json={"data": []})
        if port == 11434 and path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "c"}]})
        if port == 11434 and path == "/api/version":
            return httpx.Response(200, json={"version": "0.9"})
        raise httpx.ConnectError("refused", request=request)

    _mock_connection(monkeypatch, respond)
    assert (
        client.put(
            f"{PREFIX}/providers/mine",
            json={"kind": "openai", "base_url": "http://127.0.0.1:11434/v1"},
        ).status_code
        == 200
    )
    servers = client.post(f"{PREFIX}/local-servers").json()
    assert servers == [
        {
            "base_url": "http://127.0.0.1:1235/v1",
            "server": "lmstudio",
            "models": 2,
            "provider": None,
        },
        {
            "base_url": "http://127.0.0.1:11434/v1",
            "server": "ollama",
            "models": 1,
            "provider": "mine",
        },
    ]


def test_local_server_scan_ignores_a_broken_port_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{port")
    monkeypatch.setattr(settings_api, "LOCAL_SERVER_PORTS", (1234,))
    monkeypatch.setattr(
        settings_api, "local_port_files", lambda: [broken, tmp_path / "absent.json"]
    )

    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _mock_connection(monkeypatch, respond)
    assert client.post(f"{PREFIX}/local-servers").json() == []


def test_model_probe_marks_a_dead_model_unavailable_in_the_public_config(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_health.clear()
    answers = {"remote-fast": 200, "remote-spare": 500}

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(answers[json.loads(request.content)["model"]], json={})

    _mock_connection(monkeypatch, respond)

    def profiles() -> dict[str, Any]:
        config = client.get("/api/config/public").json()
        return {item["name"]: item["available"] for item in config["llm_profiles"]}

    assert profiles() == {"fast": None, "spare": None}
    assert client.post(f"{PREFIX}/models/fast/probe").json()["ok"] is True
    assert client.post(f"{PREFIX}/models/spare/probe").json()["ok"] is False
    assert profiles() == {"fast": "ok", "spare": "failed"}
    # A new server model makes the check stale until the next probe.
    assert client.put(f"{PREFIX}/models/spare", json={"model": "remote-fast"}).status_code == 200
    assert profiles()["spare"] is None
    model_health.clear()


def test_model_probe_rejects_an_unknown_model(client: TestClient) -> None:
    assert client.post(f"{PREFIX}/models/absent/probe").status_code == 404
