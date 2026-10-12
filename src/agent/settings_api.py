"""Expose authenticated, secret-free model settings and connection checks."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from agent.config import Settings, get_settings
from agent.database import get_db
from agent.events import emit_event
from agent.model_config import ModelConfigError, ModelId, PricingSpec, ReasoningSpec
from agent.model_overrides import (
    EffectiveModels,
    LLMModelOverride,
    LLMProviderOverride,
    LLMRoutingOverride,
    ModelLayers,
    OverrideEntry,
    OverrideSnapshot,
    SecretKeyError,
    encrypt_secret,
    load_layers,
)
from agent.models import OutboundAudit, Session


def _safe_errors(error: ValidationError | RequestValidationError) -> list[dict[str, Any]]:
    return [{"loc": item["loc"], "msg": item["msg"]} for item in error.errors()]


class _SettingsRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def safe_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as error:
                raise HTTPException(422, detail=_safe_errors(error)) from None

        return safe_handler


router = APIRouter(
    prefix="/api/settings/models", tags=["model settings"], route_class=_SettingsRoute
)
Database = Annotated[AsyncSession, Depends(get_db)]
Configuration = Annotated[Settings, Depends(get_settings)]


class _Patch(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class ProviderPatch(_Patch):
    kind: Literal["openai", "anthropic", "jev"] | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = Field(
        default=None, repr=False, exclude=True, json_schema_extra={"writeOnly": True}
    )
    timeout_seconds: float | None = None
    max_retries: int | None = None
    disabled: bool | None = None


class ModelPatch(_Patch):
    provider: str | None = None
    model: str | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    reasoning: ReasoningSpec | None = None
    pricing: PricingSpec | None = None
    disabled: bool | None = None


class RoutingPatch(_Patch):
    default: str | None = None
    fallback: list[str] | None = None
    roles: dict[str, str] | None = None
    light: str | None = None


async def _layers(db: AsyncSession, settings: Settings) -> ModelLayers:
    try:
        return await load_layers(db, settings)
    except (ValidationError, ModelConfigError):
        raise HTTPException(422, "Invalid base model configuration") from None


def _merge(layers: ModelLayers, snapshot: OverrideSnapshot | None = None) -> EffectiveModels:
    try:
        return layers.merge(snapshot)
    except ValidationError as error:
        raise HTTPException(422, detail=_safe_errors(error)) from None
    except ModelConfigError:
        raise HTTPException(422, "Invalid model configuration") from None
    except SecretKeyError as error:
        raise HTTPException(503, str(error)) from None


def _disabled(fields: dict[str, Any], previous: OverrideEntry) -> bool:
    value = fields.pop("disabled", previous.disabled)
    if not isinstance(value, bool):
        raise HTTPException(422, "disabled must be a boolean")
    return value


async def _keep_session_models(
    db: AsyncSession, current: EffectiveModels, proposed: EffectiveModels
) -> None:
    """Refuse a change that removes a model which a non-archived session uses."""
    removed = set(current.registry.models) - set(proposed.registry.models)
    if not removed:
        return
    in_use = (
        await db.scalars(
            select(Session.llm_profile)
            .where(Session.archived.is_(False), Session.llm_profile.in_(removed))
            .distinct()
        )
    ).all()
    if in_use:
        names = ", ".join(sorted(in_use))
        raise HTTPException(409, f"Sessions use these models; archive them first: {names}")


async def _changed(db: AsyncSession, effective: EffectiveModels) -> dict[str, Any]:
    await emit_event(db, "settings.models.changed", {"checksum": effective.registry.checksum})
    return effective.view


@router.get("")
async def get_models(db: Database, settings: Configuration) -> dict[str, Any]:
    return _merge(await _layers(db, settings)).view


@router.put("/providers/{provider_id}")
async def put_provider(
    provider_id: ModelId, body: ProviderPatch, db: Database, settings: Configuration
) -> dict[str, Any]:
    layers = await _layers(db, settings)
    previous = layers.overrides.providers.get(provider_id, OverrideEntry())
    fields = body.model_dump(exclude_unset=True)
    disabled = _disabled(fields, previous)
    ciphertext = previous.api_key_ciphertext
    if {"api_key", "api_key_env"} <= body.model_fields_set:
        raise HTTPException(422, "Specify either api_key or api_key_env, not both")
    if "api_key_env" in body.model_fields_set:
        ciphertext = None
    if "api_key" in body.model_fields_set:
        try:
            ciphertext = (
                encrypt_secret(body.api_key, settings, layers.env) if body.api_key else None
            )
        except SecretKeyError as error:
            raise HTTPException(503, str(error)) from None
    proposed = OverrideEntry(previous.fields | fields, disabled, ciphertext)
    snapshot = replace(
        layers.overrides, providers=layers.overrides.providers | {provider_id: proposed}
    )
    effective = _merge(layers, snapshot)
    await _keep_session_models(db, _merge(layers), effective)
    row = await db.get(LLMProviderOverride, provider_id)
    if row is None:
        row = LLMProviderOverride(id=provider_id)
        db.add(row)
    row.fields = proposed.fields
    row.disabled = proposed.disabled
    row.api_key_ciphertext = proposed.api_key_ciphertext
    return await _changed(db, effective)


@router.put("/models/{model_id}")
async def put_model(
    model_id: ModelId, body: ModelPatch, db: Database, settings: Configuration
) -> dict[str, Any]:
    layers = await _layers(db, settings)
    previous = layers.overrides.models.get(model_id, OverrideEntry())
    fields = body.model_dump(exclude_unset=True)
    disabled = _disabled(fields, previous)
    if body.reasoning is not None:
        fields["reasoning"] = body.reasoning.model_dump()
    if body.pricing is not None:
        fields["pricing"] = body.pricing.model_dump()
    proposed = OverrideEntry(previous.fields | fields, disabled)
    snapshot = replace(layers.overrides, models=layers.overrides.models | {model_id: proposed})
    effective = _merge(layers, snapshot)
    await _keep_session_models(db, _merge(layers), effective)
    row = await db.get(LLMModelOverride, model_id)
    if row is None:
        row = LLMModelOverride(id=model_id)
        db.add(row)
    row.fields = proposed.fields
    row.disabled = proposed.disabled
    return await _changed(db, effective)


@router.put("/routing")
async def put_routing(body: RoutingPatch, db: Database, settings: Configuration) -> dict[str, Any]:
    layers = await _layers(db, settings)
    proposed = layers.overrides.routing | body.model_dump(exclude_unset=True)
    effective = _merge(layers, replace(layers.overrides, routing=proposed))
    await _keep_session_models(db, _merge(layers), effective)
    row = await db.get(LLMRoutingOverride, 1)
    if row is None:
        row = LLMRoutingOverride(id=1)
        db.add(row)
    row.fields = proposed
    return await _changed(db, effective)


async def _delete_element(
    category: Literal["providers", "models"],
    element_id: str,
    disable: bool,
    db: AsyncSession,
    settings: Settings,
) -> dict[str, Any]:
    layers = await _layers(db, settings)
    current = _merge(layers)
    if not any(item["id"] == element_id for item in current.view[category]):
        raise HTTPException(404, "Unknown model settings ID")
    entries = dict(getattr(layers.overrides, category))
    if disable:
        entries[element_id] = replace(entries.get(element_id, OverrideEntry()), disabled=True)
    else:
        entries.pop(element_id, None)
    snapshot = (
        replace(layers.overrides, providers=entries)
        if category == "providers"
        else replace(layers.overrides, models=entries)
    )
    effective = _merge(layers, snapshot)
    await _keep_session_models(db, _merge(layers), effective)
    row: LLMProviderOverride | LLMModelOverride | None
    if category == "providers":
        row = await db.get(LLMProviderOverride, element_id)
    else:
        row = await db.get(LLMModelOverride, element_id)
    if disable:
        if row is None:
            row = (
                LLMProviderOverride(id=element_id, fields={})
                if category == "providers"
                else LLMModelOverride(id=element_id, fields={})
            )
            db.add(row)
        row.disabled = True
    elif row is not None:
        await db.delete(row)
    return await _changed(db, effective)


@router.delete("/providers/{provider_id}")
async def delete_provider(
    provider_id: ModelId, db: Database, settings: Configuration, disable: bool = False
) -> dict[str, Any]:
    return await _delete_element("providers", provider_id, disable, db, settings)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: ModelId, db: Database, settings: Configuration, disable: bool = False
) -> dict[str, Any]:
    return await _delete_element("models", model_id, disable, db, settings)


def connection_client() -> httpx.AsyncClient:
    """Create an isolated client with a bounded timeout and no redirects."""
    return httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False)


@router.post("/providers/{provider_id}/test")
async def test_provider_connection(
    provider_id: ModelId, db: Database, settings: Configuration
) -> dict[str, Any]:
    effective = _merge(await _layers(db, settings))
    provider = effective.config.providers.get(provider_id)
    if provider is None:
        raise HTTPException(404, "Unknown or disabled provider")
    key = effective.provider_keys[provider_id]
    if provider.api_key_env is not None and not key:
        return {"ok": False, "status": None, "models": [], "message": "API key is not configured"}
    destination = f"{provider.base_url.rstrip('/')}/models"
    audit = OutboundAudit(
        category="llm",
        destination=destination,
        operation="list_models",
        payload_bytes=0,
        payload_sha256=hashlib.sha256(b"").hexdigest(),
        status="started",
    )
    db.add(audit)
    await db.commit()
    result: dict[str, Any] = {"ok": False, "status": None, "models": []}
    try:
        async with connection_client() as client:
            response = await client.get(
                destination, headers={"Authorization": f"Bearer {key}"} if key else {}
            )
        result["status"] = response.status_code
        if response.is_success:
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError("Invalid model listing")
            result["models"] = [
                item["id"]
                for item in data["data"]
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and (not key or key not in item["id"])
            ]
            result["ok"] = True
        else:
            result["message"] = "Provider returned a non-success HTTP status"
    except (httpx.HTTPError, httpx.InvalidURL):
        result["message"] = "Cannot connect to provider within the request timeout"
    except (ValueError, UnicodeError):
        result["message"] = "Provider returned an invalid model listing"
    audit.status = "completed" if result["ok"] else "error"
    audit.detail = str(result["status"]) if result["status"] is not None else result["message"]
    await db.commit()
    return result
