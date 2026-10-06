"""Exercise model layering, credential precedence and embedded key storage."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from cryptography.fernet import Fernet
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent.config import PROJECT_ROOT, Settings
from agent.model_config import ModelsConfig, load_models
from agent.model_overrides import (
    LLMModelOverride,
    OverrideEntry,
    OverrideSnapshot,
    SecretKeyError,
    decrypt_secret,
    encrypt_secret,
    environment,
    load_effective_models,
    merge_layers,
)
from agent.models import Base


@pytest.fixture
def base() -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "providers": {
                "cloud": {
                    "kind": "openai",
                    "base_url": "https://example.com/v1",
                    "api_key_env": "CLOUD_KEY",
                    "timeout_seconds": 45,
                    "max_retries": 3,
                },
                "idle": {"kind": "openai", "base_url": "http://localhost:1234/v1"},
            },
            "models": {
                "fast": {
                    "provider": "cloud",
                    "model": "remote-fast",
                    "context_window": 16000,
                    "max_tokens": 2000,
                    "reasoning": {
                        "supported": True,
                        "allowed_efforts": ["low", "high"],
                        "default_effort": "low",
                        "wire_parameter": "reasoning_effort",
                    },
                    "pricing": {"input_per_mtok": 1, "output_per_mtok": 2},
                },
                "spare": {
                    "provider": "cloud",
                    "model": "remote-spare",
                    "context_window": 8000,
                    "max_tokens": 1000,
                },
            },
            "routing": {"default": "fast", "fallback": [], "roles": {"review": "fast"}},
        }
    )


def test_reasoning_override_is_atomic_and_other_fields_keep_file_sources(
    base: ModelsConfig,
) -> None:
    snapshot = OverrideSnapshot(models={"fast": OverrideEntry({"reasoning": {"supported": False}})})
    effective = merge_layers(base, None, snapshot, {"CLOUD_KEY": "original-abcd"})
    model = effective.registry.get("fast")
    assert model.context_window == 16000
    assert model.max_tokens == 2000
    assert model.model == "remote-fast"
    assert model.reasoning.supported is False
    assert model.reasoning.allowed_efforts == []
    assert model.reasoning.default_effort is None
    assert model.reasoning.wire_parameter is None
    assert model.pricing == base.models["fast"].pricing
    view = effective.view["models"][0]
    assert view["origin"] == "ui"
    assert view["sources"]["reasoning"] == "ui"
    assert view["sources"]["context_window"] == "file"
    assert view["sources"]["provider"] == "file"
    assert effective.view["providers"][0]["sources"]["api_key"] == "env"


def test_disabled_elements_stay_visible_but_are_excluded_from_runtime(base: ModelsConfig) -> None:
    overrides = OverrideSnapshot(
        providers={"idle": OverrideEntry(disabled=True)},
        models={"spare": OverrideEntry(disabled=True)},
    )
    result = merge_layers(base, None, overrides, {})
    assert "spare" not in result.registry.models
    assert "idle" not in result.config.providers
    assert result.view["models"][1]["disabled"] is True
    assert result.view["providers"][1]["disabled"] is True


def test_disabling_referenced_model_is_rejected(base: ModelsConfig) -> None:
    with pytest.raises(ValidationError):
        merge_layers(
            base, None, OverrideSnapshot(models={"fast": OverrideEntry(disabled=True)}), {}
        )


def test_new_ui_model_and_reset_to_file(base: ModelsConfig) -> None:
    original = base.models["spare"].model_dump()
    overrides = OverrideSnapshot(
        models={
            "custom": OverrideEntry(original | {"model": "ui-custom"}),
            "fast": OverrideEntry({"max_tokens": 3000}),
        }
    )
    effective = merge_layers(base, None, overrides, {})
    assert effective.registry.get("custom").model == "ui-custom"
    assert effective.registry.get("fast").max_tokens == 3000
    assert effective.view["models"][-1]["origin"] == "ui"
    assert set(effective.view["models"][-1]["sources"].values()) == {"ui"}
    reset = merge_layers(base, None, replace(overrides, models={}), {})
    assert reset.registry.get("fast").max_tokens == 2000
    assert "custom" not in reset.registry.models
    assert reset.view["models"][0]["sources"]["max_tokens"] == "file"
    assert reset.registry.checksum != effective.registry.checksum


def test_provider_field_and_credential_precedence(base: ModelsConfig, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    env = {"CLOUD_KEY": "file-secret-abcd", "OTHER_KEY": "other-secret-wxyz"}
    encrypted = encrypt_secret("ui-secret-1234", settings, env)
    overrides = OverrideSnapshot(
        providers={
            "cloud": OverrideEntry(
                {"base_url": "https://other.example/v1", "api_key_env": "OTHER_KEY"},
                api_key_ciphertext=encrypted,
            )
        }
    )
    result = merge_layers(base, None, overrides, env, settings=settings)
    assert result.registry.get("fast").api_key == "ui-secret-1234"
    assert result.registry.get("fast").timeout_seconds == 45
    assert result.registry.get("fast").max_retries == 3
    assert result.view["providers"][0]["api_key_hint"] == "…1234"
    assert result.view["providers"][0]["sources"]["api_key"] == "ui"
    assert result.view["providers"][0]["sources"]["base_url"] == "ui"
    assert result.view["providers"][0]["sources"]["timeout_seconds"] == "file"
    assert "ui-secret-1234" not in json.dumps(result.view)
    assert encrypted not in json.dumps(result.view)
    no_cipher = replace(
        overrides,
        providers={"cloud": replace(overrides.providers["cloud"], api_key_ciphertext=None)},
    )
    assert merge_layers(base, None, no_cipher, env).registry.get("fast").api_key == env["OTHER_KEY"]
    assert (
        merge_layers(base, None, OverrideSnapshot(), env).registry.get("fast").api_key
        == (env["CLOUD_KEY"])
    )


def test_short_key_never_appears_in_full_in_hint(base: ModelsConfig) -> None:
    result = merge_layers(base, None, OverrideSnapshot(), {"CLOUD_KEY": "abc"})
    assert result.view["providers"][0]["api_key_hint"] == "…"


def test_routing_ui_beats_env_and_preserves_other_fields(base: ModelsConfig) -> None:
    overrides = OverrideSnapshot(routing={"default": "spare", "fallback": ["fast"]})
    result = merge_layers(base, None, overrides, {"AGENT_DEFAULT_LLM_PROFILE": "fast"})
    assert result.registry.default_id == "spare"
    assert result.registry.fallback == ["fast"]
    assert result.registry.roles == {"review": "fast"}
    assert result.view["routing"]["sources"] == {"default": "ui", "fallback": "ui", "roles": "file"}
    env_result = merge_layers(
        base, None, OverrideSnapshot(), {"AGENT_DEFAULT_LLM_PROFILE": "spare"}
    )
    assert env_result.view["routing"]["sources"]["default"] == "env"


def test_invalid_merged_model_and_routing_are_rejected(base: ModelsConfig) -> None:
    with pytest.raises(ValidationError):
        merge_layers(
            base, None, OverrideSnapshot(models={"fast": OverrideEntry({"max_tokens": 16000})}), {}
        )
    with pytest.raises(ValidationError):
        merge_layers(base, None, OverrideSnapshot(routing={"default": "absent"}), {})


def test_legacy_base_preserves_credentials_and_provenance(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, llm_api_key="legacy-secret-abcd")
    legacy = load_models(None, {}, settings)
    result = merge_layers(
        None, legacy, OverrideSnapshot(models={"default": OverrideEntry({"max_tokens": 4000})}), {}
    )
    assert result.registry.source == "legacy"
    assert result.registry.get("default").api_key == "legacy-secret-abcd"
    assert result.registry.get("default").model == settings.llm_model
    assert result.registry.get("default").max_tokens == 4000
    assert result.view["models"][0]["sources"]["model"] == "legacy"
    assert result.view["models"][0]["sources"]["max_tokens"] == "ui"
    assert result.view["providers"][0]["sources"]["api_key"] == "legacy"
    assert "legacy-secret-abcd" not in json.dumps(result.view)


def test_embedded_secret_is_encrypted_and_master_key_is_private(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    token = encrypt_secret("private-secret", settings, {})
    assert "private-secret" not in token
    assert decrypt_secret(token, settings, {}) == "private-secret"
    assert (tmp_path / "secret.key").stat().st_mode & 0o777 == 0o600
    assert encrypt_secret("private-secret", settings, {}) != token


def test_concurrent_embedded_key_creation_uses_one_master_key(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(lambda _: encrypt_secret("private-secret", settings, {}), range(8)))
    assert all(decrypt_secret(token, settings, {}) == "private-secret" for token in tokens)


def test_redis_requires_shared_key_and_does_not_create_local_key(tmp_path: Path) -> None:
    embedded = Settings(_env_file=None, data_dir=tmp_path)
    redis = Settings(_env_file=None, data_dir=tmp_path, execution_mode="redis")
    token = encrypt_secret("private-secret", embedded, {})
    with pytest.raises(SecretKeyError, match="AGENT_SECRET_KEY.*redis"):
        decrypt_secret(token, redis, {})
    with pytest.raises(SecretKeyError, match="AGENT_SECRET_KEY.*redis"):
        encrypt_secret("private-secret", redis, {})
    env = {"AGENT_SECRET_KEY": Fernet.generate_key().decode()}
    shared = encrypt_secret("private-secret", redis, env)
    assert decrypt_secret(shared, redis, env) == "private-secret"


def test_invalid_or_lost_master_key_is_safe_and_never_regenerated(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    with pytest.raises(SecretKeyError, match="32-byte") as error:
        encrypt_secret("private-secret", settings, {"AGENT_SECRET_KEY": "invalid-master-secret"})
    assert "invalid-master-secret" not in str(error.value)
    token = encrypt_secret("private-secret", settings, {})
    with pytest.raises(SecretKeyError, match="Cannot decrypt"):
        decrypt_secret(token, settings, {"AGENT_SECRET_KEY": Fernet.generate_key().decode()})
    (tmp_path / "secret.key").unlink()
    with pytest.raises(SecretKeyError, match="restore"):
        decrypt_secret(token, settings, {})
    assert not (tmp_path / "secret.key").exists()


@pytest.mark.parametrize("suffix", ["!", "=", "x"])
def test_env_master_key_requires_canonical_urlsafe_base64(tmp_path: Path, suffix: str) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    invalid = Fernet.generate_key().decode() + suffix
    with pytest.raises(SecretKeyError, match="32-byte"):
        encrypt_secret("private-secret", settings, {"AGENT_SECRET_KEY": invalid})
    assert not (tmp_path / "secret.key").exists()


def test_model_override_migration_upgrade_and_downgrade() -> None:
    path = PROJECT_ROOT / "migrations" / "versions" / "0006_model_overrides.py"
    spec = spec_from_file_location("model_overrides_migration_test", path)
    assert spec is not None and spec.loader is not None
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "0005"
    engine = create_engine("sqlite://")
    try:
        with (
            engine.begin() as connection,
            Operations.context(MigrationContext.configure(connection)),
        ):
            migration.upgrade()
            schema = inspect(connection)
            assert set(schema.get_table_names()) == {
                "llm_provider_overrides",
                "llm_model_overrides",
                "llm_routing_override",
            }
            assert schema.get_pk_constraint("llm_provider_overrides")["constrained_columns"] == [
                "id"
            ]
            assert schema.get_check_constraints("llm_routing_override")[0]["sqltext"] == "id = 1"
            migration.downgrade()
            assert inspect(connection).get_table_names() == []
    finally:
        engine.dispose()


def test_environment_dotenv_then_process_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("CUSTOM_MODEL_KEY=dotenv-secret\nDOTENV_ONLY=yes\n")
    monkeypatch.setenv("AGENT_ENV_FILE", str(path))
    monkeypatch.setenv("CUSTOM_MODEL_KEY", "process-secret")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    env = environment(settings)
    assert env["CUSTOM_MODEL_KEY"] == "process-secret"
    assert env["DOTENV_ONLY"] == "yes"


async def test_database_loader_supports_yaml_with_unset_base_key(
    base: ModelsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps(base.model_dump()))
    settings = Settings(_env_file=None, data_dir=tmp_path)
    monkeypatch.delenv("CLOUD_KEY", raising=False)
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine)() as db:
            db.add(LLMModelOverride(id="fast", fields={"max_tokens": 3000}))
            await db.commit()
            result = await load_effective_models(db, settings, path)
            assert result.registry.get("fast").max_tokens == 3000
            assert not result.registry.get("fast").configured
            assert "environment variable is missing" in result.registry.warnings[0]
    finally:
        await engine.dispose()


async def test_database_loader_without_yaml_uses_legacy(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, llm_api_key="legacy-secret-abcd")
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine)() as db:
            result = await load_effective_models(db, settings, tmp_path / "missing.yaml")
            assert result.registry.source == "legacy"
            assert result.registry.get("default").api_key == "legacy-secret-abcd"
    finally:
        await engine.dispose()


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_empty_master_key_uses_the_local_key_file(tmp_path: Path, value: str) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    token = encrypt_secret("private-secret", settings, {"AGENT_SECRET_KEY": value})
    assert (tmp_path / "secret.key").exists()
    assert decrypt_secret(token, settings, {"AGENT_SECRET_KEY": value}) == "private-secret"
