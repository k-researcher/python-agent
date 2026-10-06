"""Merge declarative model settings with encrypted database overrides."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from agent import model_config
from agent.config import Settings
from agent.model_config import (
    ModelConfigError,
    ModelsConfig,
    ModelSpec,
    ProviderSpec,
    ResolvedModel,
    ResolvedModelRegistry,
)
from agent.model_registry import models_source, process_env
from agent.models import Base, utcnow

Source = Literal["file", "ui", "env", "legacy"]
Origin = Literal["file", "ui", "legacy"]


class LLMProviderOverride(Base):
    """Store provider deltas without plaintext credentials."""

    __tablename__ = "llm_provider_overrides"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    fields: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    api_key_ciphertext: Mapped[str | None] = mapped_column(Text, nullable=True)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class LLMModelOverride(Base):
    """Store model deltas with atomic nested values."""

    __tablename__ = "llm_model_overrides"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    fields: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class LLMRoutingOverride(Base):
    """Store the singleton routing delta."""

    __tablename__ = "llm_routing_override"
    __table_args__ = (CheckConstraint("id = 1", name="ck_llm_routing_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    fields: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class SecretKeyError(RuntimeError):
    """Report credential failures without secret values."""


def environment(settings: Settings) -> dict[str, str]:
    """Read dotenv values, then apply the process environment."""
    return process_env()


def _local_key(settings: Settings, *, create: bool) -> bytes:
    path = settings.data_dir / "secret.key"
    if create and not path.exists():
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".secret-", dir=settings.data_dir)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(Fernet.generate_key())
                stream.flush()
                os.fsync(stream.fileno())
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            os.unlink(temporary)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise SecretKeyError("secret.key must be a regular file")
            os.fchmod(stream.fileno(), 0o600)
            return stream.read(128).strip()
    except OSError:
        raise SecretKeyError("Cannot read secret.key; restore the original master key") from None


def _cipher(settings: Settings, env: Mapping[str, str] | None, *, create: bool) -> Fernet:
    values = environment(settings) if env is None else env
    key = (values.get("AGENT_SECRET_KEY") or "").strip() or None  # empty means "not set"
    if key is not None:
        try:
            encoded = key.encode("ascii")
            decoded = base64.b64decode(encoded, altchars=b"-_", validate=True)
            if len(decoded) != 32 or base64.urlsafe_b64encode(decoded) != encoded:
                raise ValueError("Invalid master key")
            return Fernet(encoded)
        except (ValueError, UnicodeError):
            raise SecretKeyError("AGENT_SECRET_KEY must be a urlsafe base64 32-byte key") from None
    if settings.execution_mode != "embedded":
        raise SecretKeyError("AGENT_SECRET_KEY is required for encrypted keys in redis mode")
    try:
        return Fernet(_local_key(settings, create=create))
    except ValueError:
        raise SecretKeyError("secret.key is invalid; restore the original master key") from None


def encrypt_secret(secret: str, settings: Settings, env: Mapping[str, str] | None = None) -> str:
    """Encrypt a credential using the shared or embedded master key."""
    return _cipher(settings, env, create=True).encrypt(secret.encode()).decode("ascii")


def decrypt_secret(
    ciphertext: str, settings: Settings, env: Mapping[str, str] | None = None
) -> str:
    """Decrypt a credential without creating replacement master keys."""
    cipher = _cipher(settings, env, create=False)
    try:
        return cipher.decrypt(ciphertext.encode("ascii")).decode()
    except (InvalidToken, UnicodeError):
        raise SecretKeyError(
            "Cannot decrypt API key; check the original AGENT_SECRET_KEY"
        ) from None


@dataclass(frozen=True, slots=True)
class OverrideEntry:
    fields: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    api_key_ciphertext: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class OverrideSnapshot:
    providers: dict[str, OverrideEntry] = field(default_factory=dict)
    models: dict[str, OverrideEntry] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)


async def read_overrides(db: AsyncSession) -> OverrideSnapshot:
    """Read one transaction's provider, model and routing overrides."""
    providers = {
        row.id: OverrideEntry(dict(row.fields), row.disabled, row.api_key_ciphertext)
        for row in (await db.scalars(select(LLMProviderOverride))).all()
    }
    models = {
        row.id: OverrideEntry(dict(row.fields), row.disabled)
        for row in (await db.scalars(select(LLMModelOverride))).all()
    }
    routing = await db.get(LLMRoutingOverride, 1)
    return OverrideSnapshot(providers, models, dict(routing.fields) if routing else {})


def _legacy_config(registry: ResolvedModelRegistry) -> ModelsConfig:
    providers: dict[str, ProviderSpec] = {}
    models: dict[str, ModelSpec] = {}
    for model_id, model in registry.models.items():
        providers[model.provider_id] = ProviderSpec(
            kind=model.kind,
            base_url=model.base_url,
            timeout_seconds=model.timeout_seconds,
            max_retries=model.max_retries,
        )
        models[model_id] = ModelSpec(
            provider=model.provider_id,
            model=model.model,
            context_window=model.context_window,
            max_tokens=model.max_tokens,
            reasoning=model.reasoning,
            pricing=model.pricing,
        )
    return ModelsConfig.model_validate(
        {
            "providers": providers,
            "models": models,
            "routing": {
                "default": registry.default_id,
                "fallback": registry.fallback,
                "roles": registry.roles,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class EffectiveModels:
    registry: ResolvedModelRegistry = field(repr=False)
    config: ModelsConfig
    view: dict[str, Any]
    provider_keys: dict[str, str] = field(repr=False)


def merge_layers(
    base: ModelsConfig | None,
    legacy_registry: ResolvedModelRegistry | None,
    overrides: OverrideSnapshot,
    env: Mapping[str, str],
    *,
    settings: Settings | None = None,
) -> EffectiveModels:
    """Validate fieldwise deltas and build a runtime registry and secret-free view."""
    if base is None and legacy_registry is None:
        raise ModelConfigError("No YAML configuration or legacy registry provided")
    origin: Origin = "file" if base is not None else "legacy"
    if base is None:
        assert legacy_registry is not None
        base = _legacy_config(legacy_registry)
    provider_keys: dict[str, str] = {}
    provider_configured: dict[str, bool] = {}
    legacy_keys = (
        {model.provider_id: model.api_key for model in legacy_registry.models.values()}
        if origin == "legacy" and legacy_registry is not None
        else {}
    )
    warnings = list(legacy_registry.warnings) if origin == "legacy" and legacy_registry else []
    providers: dict[str, ProviderSpec] = {}
    models: dict[str, ModelSpec] = {}
    provider_views: list[dict[str, Any]] = []
    model_views: list[dict[str, Any]] = []
    for provider_id in dict.fromkeys([*base.providers, *overrides.providers]):
        original_provider = base.providers.get(provider_id)
        delta = overrides.providers.get(provider_id, OverrideEntry())
        spec = ProviderSpec.model_validate(
            (original_provider.model_dump() if original_provider else {}) | delta.fields
        )
        sources: dict[str, Source] = {
            name: "ui" if name in delta.fields or original_provider is None else origin
            for name in ProviderSpec.model_fields
        }
        if delta.api_key_ciphertext is not None:
            if settings is None:
                raise SecretKeyError("Settings are required to decrypt UI API keys")
            key = decrypt_secret(delta.api_key_ciphertext, settings, env)
            sources["api_key"] = "ui"
        elif spec.api_key_env is not None:
            key = env.get(spec.api_key_env, "")
            sources["api_key"] = "env"
        elif "api_key_env" in delta.fields:
            key = ""
            sources["api_key"] = "ui"
        else:
            key = legacy_keys.get(provider_id, "")
            sources["api_key"] = "legacy" if origin == "legacy" else "env"
        provider_keys[provider_id] = key
        provider_configured[provider_id] = (
            bool(key)
            if origin == "legacy"
            and provider_id in legacy_keys
            and "api_key_env" not in delta.fields
            and delta.api_key_ciphertext is None
            else spec.api_key_env is None or bool(key)
        )
        if not delta.disabled:
            providers[provider_id] = spec
            if spec.api_key_env and not key:
                warnings.append(f"Provider {provider_id}: API key environment variable is missing")
            parsed = urlsplit(spec.base_url)
            if parsed.scheme == "http" and parsed.hostname not in model_config._LOCAL_HOSTS:
                warnings.append(f"Provider {provider_id}: non-local HTTP base URL is not encrypted")
        provider_views.append(
            {
                "id": provider_id,
                **spec.model_dump(),
                "api_key_set": bool(key),
                "api_key_hint": f"…{key[-4:]}" if len(key) > 4 else "…" if key else "",
                "configured": provider_configured[provider_id],
                "disabled": delta.disabled,
                "origin": "ui" if provider_id in overrides.providers else origin,
                "sources": sources
                | {"disabled": "ui" if provider_id in overrides.providers else origin},
            }
        )
    for model_id in dict.fromkeys([*base.models, *overrides.models]):
        original_model = base.models.get(model_id)
        delta = overrides.models.get(model_id, OverrideEntry())
        spec_model = ModelSpec.model_validate(
            (original_model.model_dump() if original_model else {}) | delta.fields
        )
        if not delta.disabled:
            models[model_id] = spec_model
        model_views.append(
            {
                "id": model_id,
                **spec_model.model_dump(),
                "disabled": delta.disabled,
                "origin": "ui" if model_id in overrides.models else origin,
                "sources": {
                    name: "ui" if name in delta.fields or original_model is None else origin
                    for name in ModelSpec.model_fields
                }
                | {"disabled": "ui" if model_id in overrides.models else origin},
            }
        )
    routing = base.routing.model_dump()
    routing_sources: dict[str, Source] = dict.fromkeys(routing, origin)
    env_default = env.get("AGENT_DEFAULT_LLM_PROFILE")
    if origin == "file" and env_default is not None and env_default != routing["default"]:
        if env_default in models:
            routing["default"] = env_default
            routing_sources["default"] = "env"
        else:
            warnings.append(
                "AGENT_DEFAULT_LLM_PROFILE names no model in the YAML file and is ignored"
            )
    routing.update(overrides.routing)
    routing_sources.update(dict.fromkeys(overrides.routing, "ui"))
    config = ModelsConfig.model_validate(
        {
            "providers": providers,
            "models": models,
            "routing": routing,
            "decisions": base.decisions,
        }
    )
    view: dict[str, Any] = {
        "providers": provider_views,
        "models": model_views,
        "routing": config.routing.model_dump() | {"sources": routing_sources},
        "warnings": warnings,
    }
    fingerprint = {
        "view": view,
        "decisions": config.decisions.model_dump(),
        "encrypted_keys": {
            name: delta.api_key_ciphertext for name, delta in overrides.providers.items()
        },
    }
    checksum = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    view["checksum"] = checksum
    resolved: dict[str, ResolvedModel] = {}
    for model_id, spec_model in config.models.items():
        provider = config.providers[spec_model.provider]
        key = provider_keys[spec_model.provider]
        resolved[model_id] = ResolvedModel(
            id=model_id,
            provider_id=spec_model.provider,
            kind=provider.kind,
            base_url=provider.base_url,
            api_key=key,
            configured=provider_configured[spec_model.provider],
            model=spec_model.model,
            context_window=spec_model.context_window,
            max_tokens=spec_model.max_tokens,
            timeout_seconds=provider.timeout_seconds,
            max_retries=provider.max_retries,
            reasoning=spec_model.reasoning,
            pricing=spec_model.pricing,
        )
    registry = ResolvedModelRegistry(
        models=resolved,
        default_id=config.routing.default,
        fallback=list(config.routing.fallback),
        roles=dict(config.routing.roles),
        warnings=warnings,
        source="yaml" if origin == "file" else "legacy",
        checksum=checksum,
        light_id=config.routing.light,
    )
    return EffectiveModels(registry, config, view, provider_keys)


@dataclass(frozen=True, slots=True)
class ModelLayers:
    base: ModelsConfig | None
    legacy: ResolvedModelRegistry | None = field(repr=False)
    overrides: OverrideSnapshot
    env: dict[str, str] = field(repr=False)
    settings: Settings = field(repr=False)

    def merge(self, overrides: OverrideSnapshot | None = None) -> EffectiveModels:
        """Validate the current or proposed snapshot."""
        return merge_layers(
            self.base,
            self.legacy,
            self.overrides if overrides is None else overrides,
            self.env,
            settings=self.settings,
        )


async def load_layers(
    db: AsyncSession, settings: Settings, models_file: Path | None = None
) -> ModelLayers:
    """Load the YAML structure independently of unresolved credentials."""
    env = environment(settings)
    path = models_file or models_source(settings)
    legacy: ResolvedModelRegistry | None
    try:
        if path is None:
            raise FileNotFoundError
        with path.open("rb") as stream:
            content = stream.read(1024 * 1024 + 1)
    except FileNotFoundError:
        base = None
        legacy = model_config.load_models(None, env, settings)
    except OSError:
        raise ModelConfigError("Cannot read models configuration") from None
    else:
        if len(content) > 1024 * 1024:
            raise ModelConfigError("Models configuration exceeds the 1 MiB limit")
        data = model_config._parse_yaml(content, model_config._Diagnostics(str(path)))
        base = ModelsConfig.model_validate(data)
        legacy = None
    return ModelLayers(base, legacy, await read_overrides(db), env, settings)


async def load_effective_models(
    db: AsyncSession, settings: Settings, models_file: Path | None = None
) -> EffectiveModels:
    """Return the runtime registry together with UI provenance."""
    return (await load_layers(db, settings, models_file)).merge()


async def load_effective_registry(db: AsyncSession, settings: Settings) -> ResolvedModelRegistry:
    """Return the merged registry for existing runtime consumers."""
    return (await load_effective_models(db, settings)).registry


async def load_models_view(db: AsyncSession, settings: Settings) -> dict[str, Any]:
    """Return settings without plaintext or encrypted API keys."""
    return (await load_effective_models(db, settings)).view
