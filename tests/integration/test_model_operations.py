"""Model settings stay usable when entries are disabled, removed or edited."""

from __future__ import annotations

import asyncio
import ssl
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from agent import llm, model_health, settings_api
from agent.config import Settings, get_settings
from agent.database import session_factory
from agent.model_overrides import LLMProviderOverride, load_effective_registry
from agent.models import Message, Project, Session
from tests.integration.test_run_finish_races import run_store
from tests.integration.test_settings_api import FILE_KEY, PREFIX, UI_KEY, client, model, provider

__all__ = ["client", "run_store"]


@pytest.fixture(autouse=True)
def isolate_model_health():
    model_health.clear()
    yield
    model_health.clear()


def checksum(client: TestClient) -> str:
    response = client.get(PREFIX)
    assert response.status_code == 200
    return response.json()["checksum"]


def seed_session(client: TestClient, tmp_path: Path, state: str, profile: str = "fast") -> str:
    async def seed() -> str:
        async with session_factory() as db:
            project = Project(name="history", root_path=str(tmp_path / state))
            db.add(project)
            await db.flush()
            session = Session(
                project_id=project.id,
                title="Keep the history",
                status=state,
                llm_profile=profile,
                run_generation=0,
            )
            db.add(session)
            await db.flush()
            db.add(Message(session_id=session.id, role="user", content="Saved message"))
            await db.commit()
            return session.id

    assert client.portal is not None
    return client.portal.call(seed)


def test_provider_disable_keeps_catalog_and_reenable_restores_models(client: TestClient) -> None:
    before = client.get(PREFIX).json()
    response = client.put(f"{PREFIX}/providers/cloud", json={"disabled": True})
    assert response.status_code == 200
    view = response.json()
    assert provider(view)["disabled"] is True
    assert model(view)["disabled"] is False
    assert model(view)["provider_disabled"] is True
    assert view["routing"]["default"] == ""
    assert view["routing"]["roles"] == {}
    public = client.get("/api/config/public")
    assert public.status_code == 200
    assert public.json()["llm_profiles"] == []
    assert public.json()["llm_configured"] is False
    enabled = client.put(f"{PREFIX}/providers/cloud", json={"disabled": False})
    assert enabled.status_code == 200
    assert model(enabled.json())["disabled"] is False
    assert model(enabled.json())["provider_disabled"] is False
    assert enabled.json()["routing"] == before["routing"]


@pytest.mark.parametrize("operation", ["disable", "delete"])
def test_empty_catalog_does_not_break_config_or_history(
    client: TestClient, tmp_path: Path, operation: str
) -> None:
    sid = seed_session(client, tmp_path, "completed")
    if operation == "disable":
        response = client.put(f"{PREFIX}/providers/cloud", json={"disabled": True})
    else:
        response = client.delete(f"{PREFIX}/providers/cloud")
    assert response.status_code == 200
    assert client.get(PREFIX).status_code == 200
    config = client.get("/api/config/public")
    assert config.status_code == 200
    assert config.json()["llm_configured"] is False
    assert config.json()["llm_profiles"] == []
    assert client.get(f"/api/sessions/{sid}").status_code == 200
    messages = client.get(f"/api/sessions/{sid}/messages")
    assert messages.status_code == 200
    assert any(row["content"] == "Saved message" for row in messages.json())
    assert client.get(f"/api/sessions/{sid}/context").status_code == 200


def test_deleted_file_provider_stays_deleted_until_explicit_reset(client: TestClient) -> None:
    response = client.delete(f"{PREFIX}/providers/cloud")
    assert response.status_code == 200
    for view in (response.json(), client.get(PREFIX).json()):
        assert all(item["id"] != "cloud" for item in view["providers"])
        assert all(item["provider"] != "cloud" for item in view["models"])
        assert "cloud" in view["deleted"]["providers"]

    async def read_again() -> list[str]:
        async with session_factory() as db:
            return (await load_effective_registry(db, get_settings())).ids()

    assert client.portal is not None
    assert client.portal.call(read_again) == []
    restored = client.post(f"{PREFIX}/providers/cloud/reset")
    assert restored.status_code == 200
    assert provider(restored.json())["disabled"] is False
    assert model(restored.json())["provider"] == "cloud"
    assert "cloud" not in restored.json()["deleted"]["providers"]


def test_disabled_provider_delete_does_not_reenable_file_entry(client: TestClient) -> None:
    assert client.put(f"{PREFIX}/providers/idle", json={"disabled": True}).status_code == 200
    response = client.delete(f"{PREFIX}/providers/idle")
    assert response.status_code == 200
    assert all(item["id"] != "idle" for item in response.json()["providers"])
    assert all(item["id"] != "idle" for item in client.get(PREFIX).json()["providers"])


def test_provider_reset_explicitly_restores_file_address_and_key_source(client: TestClient) -> None:
    changed = client.put(
        f"{PREFIX}/providers/cloud",
        json={"base_url": "https://other.example.com/v1", "api_key": UI_KEY, "key_source": "ui"},
    )
    assert changed.status_code == 200
    reset = client.post(f"{PREFIX}/providers/cloud/reset")
    assert reset.status_code == 200
    row = provider(reset.json())
    assert row["base_url"] == "https://example.com/v1"
    assert row["key_source"] == "env"
    assert row["api_key_hint"] == "…abcd"
    assert UI_KEY not in reset.text


def test_file_model_delete_and_reset_are_distinct(client: TestClient) -> None:
    assert client.put(f"{PREFIX}/models/fast", json={"max_tokens": 3000}).status_code == 200
    deleted = client.delete(f"{PREFIX}/models/fast")
    assert deleted.status_code == 200
    assert all(item["id"] != "fast" for item in deleted.json()["models"])
    assert "fast" in deleted.json()["deleted"]["models"]
    assert deleted.json()["routing"]["default"] == "spare"
    reset = client.post(f"{PREFIX}/models/fast/reset")
    assert reset.status_code == 200
    assert model(reset.json())["max_tokens"] == 2000
    assert reset.json()["routing"]["default"] == "fast"


@pytest.mark.parametrize("operation", ["disable", "delete"])
def test_last_model_can_be_disabled_or_deleted(client: TestClient, operation: str) -> None:
    for entry in ("fast", "spare"):
        if operation == "disable":
            response = client.put(f"{PREFIX}/models/{entry}", json={"disabled": True})
        else:
            response = client.delete(f"{PREFIX}/models/{entry}")
        assert response.status_code == 200
    public = client.get("/api/config/public")
    assert public.status_code == 200
    assert public.json()["llm_profiles"] == []
    assert public.json()["llm_configured"] is False
    assert provider(client.get(PREFIX).json())["disabled"] is False


def test_ui_only_entries_are_removed_and_cannot_reset_to_file(client: TestClient) -> None:
    assert (
        client.put(
            f"{PREFIX}/providers/new",
            json={"kind": "openai", "base_url": "http://localhost:1234/v1"},
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"{PREFIX}/models/new",
            json={"provider": "new", "model": "chat", "context_window": 8000, "max_tokens": 1000},
        ).status_code
        == 200
    )
    assert client.post(f"{PREFIX}/providers/new/reset").status_code == 404
    assert client.post(f"{PREFIX}/models/new/reset").status_code == 404
    assert client.delete(f"{PREFIX}/models/new").status_code == 200
    assert client.delete(f"{PREFIX}/providers/new").status_code == 200
    view = client.get(PREFIX).json()
    assert all(item["id"] != "new" for item in view["providers"] + view["models"])
    assert "new" not in view["deleted"]["providers"] + view["deleted"]["models"]


def test_deleting_ui_provider_removes_child_models_and_preserves_history(
    client: TestClient, tmp_path: Path
) -> None:
    assert (
        client.put(
            f"{PREFIX}/providers/new",
            json={"kind": "openai", "base_url": "http://localhost:1234/v1"},
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"{PREFIX}/models/custom",
            json={"provider": "new", "model": "chat", "context_window": 8000, "max_tokens": 1000},
        ).status_code
        == 200
    )
    sid = seed_session(client, tmp_path, "completed", "custom")
    removed = client.delete(f"{PREFIX}/providers/new")
    assert removed.status_code == 200
    view = client.get(PREFIX).json()
    assert all(item["id"] != "new" for item in view["providers"])
    assert all(item["id"] != "custom" for item in view["models"])
    assert client.get(f"/api/sessions/{sid}").json()["llm_profile"] == "custom"
    assert any(
        row["content"] == "Saved message"
        for row in client.get(f"/api/sessions/{sid}/messages").json()
    )


def test_deleting_ui_default_model_clears_its_routing_references(client: TestClient) -> None:
    assert (
        client.put(
            f"{PREFIX}/models/custom",
            json={"provider": "idle", "model": "chat", "context_window": 8000, "max_tokens": 1000},
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"{PREFIX}/routing",
            json={"default": "custom", "light": "custom", "roles": {"pm": "custom"}},
        ).status_code
        == 200
    )
    removed = client.delete(f"{PREFIX}/models/custom")
    assert removed.status_code == 200
    view = client.get(PREFIX).json()
    assert all(item["id"] != "custom" for item in view["models"])
    assert view["routing"]["default"] == "fast"
    assert view["routing"]["light"] is None
    assert "custom" not in view["routing"]["roles"].values()


@pytest.mark.parametrize("state", ["completed", "stopped", "error", "interrupted"])
def test_terminal_sessions_do_not_block_changes_or_lose_history(
    client: TestClient, tmp_path: Path, state: str
) -> None:
    sid = seed_session(client, tmp_path, state)
    response = client.put(f"{PREFIX}/models/fast", json={"disabled": True})
    assert response.status_code == 200
    assert client.get(f"/api/sessions/{sid}").json()["status"] == state
    assert any(
        row["content"] == "Saved message"
        for row in client.get(f"/api/sessions/{sid}/messages").json()
    )


@pytest.mark.parametrize("state", ["pending", "running", "awaiting_confirmation"])
@pytest.mark.parametrize(
    "change", ["provider_disable", "provider_delete", "provider_edit", "model_edit"]
)
def test_active_sessions_block_runtime_configuration_changes(
    client: TestClient, tmp_path: Path, state: str, change: str
) -> None:
    sid = seed_session(client, tmp_path, state)
    before = checksum(client)
    if change == "provider_delete":
        response = client.delete(f"{PREFIX}/providers/cloud")
    elif change == "provider_disable":
        response = client.put(f"{PREFIX}/providers/cloud", json={"disabled": True})
    elif change == "provider_edit":
        response = client.put(
            f"{PREFIX}/providers/cloud", json={"base_url": "https://other.example.com/v1"}
        )
    else:
        response = client.put(f"{PREFIX}/models/fast", json={"provider": "idle", "model": "other"})
    assert response.status_code == 409
    assert sid in response.text
    assert checksum(client) == before


def test_unavailable_session_model_cannot_silently_use_another_model(
    client: TestClient, tmp_path: Path
) -> None:
    sid = seed_session(client, tmp_path, "completed")
    assert client.put(f"{PREFIX}/models/fast", json={"disabled": True}).status_code == 200
    assert client.get("/api/config/public").json()["default_llm_profile"] == "spare"
    messages_before = client.get(f"/api/sessions/{sid}/messages").json()
    response = client.post(f"/api/sessions/{sid}/messages", json={"content": "Do not switch"})
    assert response.status_code == 409
    assert client.get(f"/api/sessions/{sid}/messages").json() == messages_before
    assert client.post(f"/api/sessions/{sid}/start").status_code == 409


def test_stale_revision_rejects_without_overwriting_newer_values(client: TestClient) -> None:
    original_retries = provider(client.get(PREFIX).json())["max_retries"]
    revision = checksum(client)
    first = client.put(
        f"{PREFIX}/providers/cloud", json={"timeout_seconds": 25}, headers={"If-Match": revision}
    )
    assert first.status_code == 200
    stale = client.put(
        f"{PREFIX}/providers/cloud", json={"max_retries": 4}, headers={"If-Match": revision}
    )
    assert stale.status_code == 409
    current = provider(client.get(PREFIX).json())
    assert current["timeout_seconds"] == 25
    assert current["max_retries"] == original_retries


@pytest.mark.parametrize("category,entry", [("providers", "cloud"), ("models", "fast")])
def test_create_only_does_not_overwrite_existing_id(
    client: TestClient, category: str, entry: str
) -> None:
    before = checksum(client)
    patch = (
        {"base_url": "https://other.example.com/v1"}
        if category == "providers"
        else {"model": "other"}
    )
    result = client.put(f"{PREFIX}/{category}/{entry}", json=patch, headers={"If-None-Match": "*"})
    assert result.status_code == 409
    assert checksum(client) == before


def test_clear_database_key_does_not_reveal_environment_fallback(client: TestClient) -> None:
    saved = client.put(f"{PREFIX}/providers/cloud", json={"api_key": UI_KEY, "key_source": "ui"})
    assert saved.status_code == 200
    assert UI_KEY not in saved.text
    cleared = client.put(f"{PREFIX}/providers/cloud", json={"api_key": ""})
    assert cleared.status_code == 200
    assert provider(cleared.json())["api_key_set"] is False
    assert provider(cleared.json())["configured"] is False
    assert provider(cleared.json())["key_source"] == "none"
    assert FILE_KEY not in cleared.text and UI_KEY not in cleared.text

    async def stored_key_is_removed() -> bool:
        async with session_factory() as db:
            row = await db.get(LLMProviderOverride, "cloud")
            return row is not None and row.api_key_ciphertext is None

    assert client.portal is not None
    assert client.portal.call(stored_key_is_removed)
    restored = client.put(f"{PREFIX}/providers/cloud", json={"key_source": "env"})
    assert restored.status_code == 200
    assert provider(restored.json())["api_key_set"] is True
    assert provider(restored.json())["sources"]["api_key"] == "env"


def test_provider_ids_with_slash_can_be_created_edited_and_deleted(client: TestClient) -> None:
    path = f"{PREFIX}/providers/team/gateway"
    assert (
        client.put(
            path, json={"kind": "openai", "base_url": "http://localhost:1234/v1"}
        ).status_code
        == 200
    )
    assert client.put(path, json={"timeout_seconds": 20}).status_code == 200
    assert client.delete(path).status_code == 200


def test_explicit_routing_typo_is_rejected_without_changes(client: TestClient) -> None:
    before = checksum(client)
    assert client.put(f"{PREFIX}/routing", json={"default": "missing"}).status_code == 422
    assert client.put(f"{PREFIX}/routing", json={"roles": {"pm": "missing"}}).status_code == 422
    assert checksum(client) == before


@pytest.mark.parametrize("field", ["default", "fallback", "roles"])
def test_nonnullable_routing_fields_reject_null(client: TestClient, field: str) -> None:
    before = checksum(client)
    assert client.put(f"{PREFIX}/routing", json={field: None}).status_code == 422
    assert checksum(client) == before


def test_probe_respects_header_auth_and_extra_headers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    path = get_settings().models_file
    assert isinstance(path, Path)
    config = yaml.safe_load(path.read_text())
    config["providers"]["cloud"].pop("api_key_env")
    config["providers"]["cloud"]["auth"] = {
        "type": "header_key",
        "key_env": "SETTINGS_TEST_KEY",
        "header": "X-Test-Key",
        "prefix": "",
    }
    config["providers"]["cloud"]["headers"] = {"X-Project": "diagnostic"}
    path.write_text(yaml.safe_dump(config))
    requests: list[dict[str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(dict(request.headers))
        assert request.headers["X-Test-Key"] == FILE_KEY
        assert request.headers["X-Project"] == "diagnostic"
        assert "Authorization" not in request.headers
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=False),
    )
    response = client.post(f"{PREFIX}/models/fast/probe")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert len(requests) == 1


def test_probe_uses_the_configured_trust_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yaml

    path = get_settings().models_file
    assert isinstance(path, Path)
    config = yaml.safe_load(path.read_text())
    bundle = str(tmp_path / "private-ca.pem")
    config["providers"]["cloud"]["ca_bundle"] = bundle
    path.write_text(yaml.safe_dump(config))
    context = ssl.create_default_context()
    verified: list[str | None] = []

    def trust_store(value: str | None) -> ssl.SSLContext:
        verified.append(value)
        return context

    def connection(*, verify: ssl.SSLContext | bool = True) -> httpx.AsyncClient:
        assert verify is context
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json={"choices": []})
            ),
            verify=verify,
            follow_redirects=False,
        )

    monkeypatch.setattr(llm, "build_ssl_context", trust_store)
    monkeypatch.setattr(settings_api, "connection_client", connection)
    response = client.post(f"{PREFIX}/models/fast/probe")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert verified == [bundle]


def test_failed_probe_is_invalidated_after_key_change(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(401))
        ),
    )
    assert client.post(f"{PREFIX}/models/fast/probe").json()["ok"] is False
    before = client.get("/api/config/public").json()
    assert (
        next(item for item in before["llm_profiles"] if item["name"] == "fast")["available"]
        == "failed"
    )
    assert client.put(f"{PREFIX}/providers/cloud", json={"api_key": UI_KEY}).status_code == 200
    after = client.get("/api/config/public").json()
    assert (
        next(item for item in after["llm_profiles"] if item["name"] == "fast")["available"] is None
    )


def test_old_probe_completion_cannot_poison_new_credentials(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        async with session_factory() as db:
            await settings_api.put_provider(
                "cloud", settings_api.ProviderPatch(api_key=UI_KEY), db, get_settings()
            )
        return httpx.Response(401)

    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    assert client.post(f"{PREFIX}/models/fast/probe").json()["ok"] is False
    config = client.get("/api/config/public").json()
    assert (
        next(item for item in config["llm_profiles"] if item["name"] == "fast")["available"] is None
    )


def test_env_key_source_updates_header_auth_reference(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    path = get_settings().models_file
    assert isinstance(path, Path)
    config = yaml.safe_load(path.read_text())
    config["providers"]["cloud"].pop("api_key_env")
    config["providers"]["cloud"]["auth"] = {
        "type": "header_key",
        "key_env": "SETTINGS_TEST_KEY",
        "header": "X-Test-Key",
        "prefix": "",
    }
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("SETTINGS_CHANGED_KEY", "changed-private-key")
    changed = client.put(
        f"{PREFIX}/providers/cloud",
        json={"key_source": "env", "api_key_env": "SETTINGS_CHANGED_KEY"},
    )
    assert changed.status_code == 200
    assert provider(changed.json())["key_env"] == "SETTINGS_CHANGED_KEY"
    assert "changed-private-key" not in changed.text

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Test-Key"] == "changed-private-key"
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(
        settings_api,
        "connection_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    assert client.post(f"{PREFIX}/models/fast/probe").json()["ok"] is True


async def test_concurrent_partial_changes_preserve_both_values(
    run_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent.model_config import ModelsConfig
    from agent.model_overrides import ModelLayers, read_overrides

    base = ModelsConfig.model_validate(
        {
            "providers": {"cloud": {"kind": "openai", "base_url": "https://example.com/v1"}},
            "models": {
                "main": {
                    "provider": "cloud",
                    "model": "chat",
                    "context_window": 8000,
                    "max_tokens": 1000,
                }
            },
            "routing": {"default": "main"},
        }
    )
    settings = Settings(_env_file=None, data_dir=tmp_path)
    async with run_store() as db:
        db.add(LLMProviderOverride(id="cloud", fields={"max_retries": 1}))
        await db.commit()

    original_layers = settings_api._layers

    async def layers(db: Any, _settings: Settings) -> Any:
        return ModelLayers(base, None, await read_overrides(db), {}, settings)

    monkeypatch.setattr(settings_api, "_layers", layers)

    async def change(patch: dict[str, Any]) -> None:
        async with run_store() as db:
            await settings_api.put_provider(
                "cloud", settings_api.ProviderPatch(**patch), db, settings
            )

    await asyncio.wait_for(
        asyncio.gather(change({"max_retries": 2}), change({"timeout_seconds": 23})), timeout=15
    )
    monkeypatch.setattr(settings_api, "_layers", original_layers)
    async with run_store() as db:
        row = await db.get(LLMProviderOverride, "cloud")
        assert row is not None
        assert row.fields["max_retries"] == 2
        assert row.fields["timeout_seconds"] == 23
