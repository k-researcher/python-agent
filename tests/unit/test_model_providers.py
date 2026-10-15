"""Flexible provider configuration: auth types, headers, templates and data residency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import agent.model_config as model_config
from agent.model_config import ModelsConfig, ProviderSpec, load_models
from agent.model_overrides import OverrideSnapshot, merge_layers

_SECRET = "provider-secret-that-must-not-leak"


def _config(provider: dict[str, Any]) -> dict[str, Any]:
    return {
        "providers": {
            "cloud": {"kind": "openai", "base_url": "https://llm.example.test/v1"} | provider
        },
        "models": {
            "main": {
                "provider": "cloud",
                "model": "assistant-pro",
                "context_window": 32000,
                "max_tokens": 4000,
            }
        },
        "routing": {"default": "main"},
    }


def _load(tmp_path: Path, provider: dict[str, Any], env: dict[str, str]) -> Any:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(_config(provider)), encoding="utf-8")
    return load_models(path, env, None)


_CLOUD = {
    "auth": {"type": "header_key", "key_env": "CLOUD_KEY", "prefix": "Api-Key "},
    "headers": {"x-folder-id": "${CLOUD_FOLDER}"},
    "model_template": "uri://${CLOUD_FOLDER}/{model}/latest",
    "data_in_russia": True,
}
_ENV = {"CLOUD_KEY": _SECRET, "CLOUD_FOLDER": "folder42"}


def test_header_key_provider_resolves_secret_headers_and_template(tmp_path: Path) -> None:
    model = _load(tmp_path, _CLOUD, _ENV).get("main")
    assert model.model == "uri://folder42/assistant-pro/latest"
    assert model.headers == {"x-folder-id": "folder42"}
    assert model.auth_secret == _SECRET
    # The legacy Bearer field stays empty: no request can send the key the old way.
    assert model.api_key == ""
    assert model.data_in_russia is True
    assert _SECRET not in repr(model)


def test_advanced_provider_stays_unconfigured_until_the_transport_supports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _load(tmp_path, _CLOUD, _ENV)
    assert registry.get("main").configured is False
    assert any("transport support" in warning for warning in registry.warnings)
    monkeypatch.setattr(model_config, "ADVANCED_PROVIDER_TRANSPORT", True)
    assert _load(tmp_path, _CLOUD, _ENV).get("main").configured is True


def test_missing_template_variable_is_named_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_config, "ADVANCED_PROVIDER_TRANSPORT", True)
    registry = _load(tmp_path, _CLOUD, {"CLOUD_KEY": _SECRET})
    assert registry.get("main").configured is False
    text = " ".join(registry.warnings)
    assert "CLOUD_FOLDER" in text
    assert _SECRET not in text


def test_missing_auth_key_makes_the_provider_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_config, "ADVANCED_PROVIDER_TRANSPORT", True)
    registry = _load(tmp_path, _CLOUD, {"CLOUD_FOLDER": "folder42"})
    assert registry.get("main").configured is False
    assert any("auth.key_env" in warning for warning in registry.warnings)


def test_oauth_command_and_none_auth_types_parse(tmp_path: Path) -> None:
    oauth = {
        "auth": {
            "type": "oauth_client_credentials",
            "token_url": "https://auth.example.test/api/v2/oauth",
            "key_env": "AUTH_KEY",
            "scope": "API_PERS",
        },
        "ca_bundle": "config/certs/root.pem",
        "dialect": "functions",
    }
    model = _load(tmp_path, oauth, {"AUTH_KEY": _SECRET}).get("main")
    assert model.auth is not None and model.auth.type == "oauth_client_credentials"
    assert model.dialect == "functions"
    assert model.ca_bundle == "config/certs/root.pem"
    command = ProviderSpec.model_validate(
        {
            "kind": "openai",
            "base_url": "https://x.test/v1",
            "auth": {"type": "command", "argv": ["tok"]},
        }
    )
    assert command.advanced is True
    plain = ProviderSpec.model_validate({"kind": "openai", "base_url": "https://x.test/v1"})
    assert plain.advanced is False


@pytest.mark.parametrize(
    "provider",
    [
        {"api_key_env": "KEY", "auth": {"type": "bearer", "key_env": "KEY"}},
        {
            "auth": {
                "type": "oauth_client_credentials",
                "token_url": "http://a.test/t",
                "key_env": "K",
                "scope": "s",
            }
        },
        {"auth": {"type": "header_key", "key_env": "1BAD"}},
        {"headers": {"bad header": "x"}},
        {"headers": {"x-ok": "line\nbreak"}},
        {"model_template": "no-placeholder"},
        {"dialect": "unknown"},
    ],
)
def test_invalid_provider_settings_are_rejected(provider: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(_config(provider))


def test_override_layer_applies_the_same_provider_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_config, "ADVANCED_PROVIDER_TRANSPORT", True)
    base = ModelsConfig.model_validate(_config(_CLOUD))
    effective = merge_layers(base, None, OverrideSnapshot(), _ENV)
    model = effective.registry.get("main")
    assert model.configured is True
    assert model.model == "uri://folder42/assistant-pro/latest"
    assert model.auth_secret == _SECRET and model.api_key == ""
    assert model.headers == {"x-folder-id": "folder42"}
    assert _SECRET not in repr(effective.view)
