"""Expose authenticated, secret-free model settings and connection checks."""

from __future__ import annotations

import hashlib
import json
import ssl
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from agent import model_config, model_health
from agent.config import Settings, get_settings
from agent.database import get_db
from agent.events import emit_event
from agent.llm import LLMClient, LLMError
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
    entry_fields,
    load_layers,
    lock_model_settings,
)
from agent.models import OutboundAudit, Session
from agent.wire import build_wire_request


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
_NO_REQUEST = cast(Request, None)


class _Patch(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class ProviderPatch(_Patch):
    kind: Literal["openai", "anthropic", "jev"] | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    key_source: Literal["ui", "env", "none"] | None = None
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
        raise HTTPException(422, "Некорректные ссылки в настройках моделей") from None
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
    """Keep active runs on the model and connection that they started with."""
    removed = {
        name
        for name, model in current.registry.models.items()
        if proposed.registry.models.get(name) != model
    }
    if not removed:
        return
    in_use = (
        await db.scalars(
            select(Session.id)
            .where(
                Session.status.in_({"pending", "running", "awaiting_confirmation"}),
                Session.llm_profile.in_(removed),
            )
            .order_by(Session.id)
            .limit(10)
        )
    ).all()
    if in_use:
        names = ", ".join(sorted(name for name in in_use if name is not None))
        raise HTTPException(409, f"Сначала остановите активные сессии: {names}")


async def _begin_change(
    db: AsyncSession, settings: Settings, request: Request | None
) -> ModelLayers:
    await lock_model_settings(db)
    layers = await _layers(db, settings)
    expected = request.headers.get("If-Match") if request is not None else None
    if expected is not None and expected.strip('"') != _merge(layers).registry.checksum:
        raise HTTPException(
            409, "Настройки изменились в другом окне. Проверьте актуальные значения."
        )
    return layers


def _check_create(
    layers: ModelLayers, category: str, element_id: str, request: Request | None
) -> None:
    if (
        request is not None
        and request.headers.get("If-None-Match") == "*"
        and any(row["id"] == element_id for row in _merge(layers).view[category])
    ):
        raise HTTPException(409, "Такое имя уже занято. Выберите другое имя.")


async def _changed(db: AsyncSession, effective: EffectiveModels) -> dict[str, Any]:
    await emit_event(db, "settings.models.changed", {"checksum": effective.registry.checksum})
    return effective.view


@router.get("")
async def get_models(db: Database, settings: Configuration) -> dict[str, Any]:
    return _merge(await _layers(db, settings)).view


@router.put("/providers/{provider_id:path}")
async def put_provider(
    provider_id: ModelId,
    body: ProviderPatch,
    db: Database,
    settings: Configuration,
    request: Request = _NO_REQUEST,
) -> dict[str, Any]:
    layers = await _begin_change(db, settings, request)
    _check_create(layers, "providers", provider_id, request)
    previous = layers.overrides.providers.get(provider_id, OverrideEntry())
    if previous.deleted:
        previous = OverrideEntry()
    fields = body.model_dump(exclude_unset=True)
    disabled = _disabled(fields, previous)
    ciphertext = previous.api_key_ciphertext
    key_source = fields.pop("key_source", previous.key_source)
    if {"api_key", "api_key_env"} <= body.model_fields_set:
        raise HTTPException(422, "Specify either api_key or api_key_env, not both")
    if "api_key_env" in body.model_fields_set:
        ciphertext = None
        key_source = "env"
        current_provider = next(
            (item for item in _merge(layers).view["providers"] if item["id"] == provider_id), None
        )
        auth = current_provider.get("auth") if current_provider else None
        if auth is not None and "key_env" in auth and body.api_key_env:
            fields["auth"] = {**auth, "key_env": body.api_key_env}
            fields["api_key_env"] = None
    if "api_key" in body.model_fields_set:
        try:
            ciphertext = (
                encrypt_secret(body.api_key, settings, layers.env) if body.api_key else None
            )
            key_source = "ui" if body.api_key else "none"
        except SecretKeyError as error:
            raise HTTPException(503, str(error)) from None
    if body.key_source == "none":
        ciphertext, key_source = None, "none"
    elif body.key_source == "env":
        ciphertext, key_source = None, "env"
    elif body.key_source == "ui" and ciphertext is None:
        raise HTTPException(422, "Введите ключ для сохранения в базе данных.")
    proposed = OverrideEntry(previous.fields | fields, disabled, ciphertext, key_source=key_source)
    snapshot = replace(
        layers.overrides, providers=layers.overrides.providers | {provider_id: proposed}
    )
    effective = _merge(layers, snapshot)
    await _keep_session_models(db, _merge(layers), effective)
    row = await db.get(LLMProviderOverride, provider_id)
    if row is None:
        row = LLMProviderOverride(id=provider_id)
        db.add(row)
    row.fields = entry_fields(proposed)
    row.disabled = proposed.disabled
    row.api_key_ciphertext = proposed.api_key_ciphertext
    return await _changed(db, effective)


# ":path" lets a model ID keep its slash, as in "publisher/model".
@router.put("/models/{model_id:path}")
async def put_model(
    model_id: ModelId,
    body: ModelPatch,
    db: Database,
    settings: Configuration,
    request: Request = _NO_REQUEST,
) -> dict[str, Any]:
    layers = await _begin_change(db, settings, request)
    _check_create(layers, "models", model_id, request)
    previous = layers.overrides.models.get(model_id, OverrideEntry())
    if previous.deleted:
        previous = OverrideEntry()
    fields = body.model_dump(exclude_unset=True)
    disabled = _disabled(fields, previous)
    if body.reasoning is not None:
        fields["reasoning"] = body.reasoning.model_dump()
    if body.pricing is not None:
        fields["pricing"] = body.pricing.model_dump()
    proposed = OverrideEntry(previous.fields | fields, disabled)
    target_provider = proposed.fields.get("provider")
    if target_provider is not None and not any(
        row["id"] == target_provider for row in _merge(layers).view["providers"]
    ):
        raise HTTPException(422, "Выберите существующий сервер модели.")
    snapshot = replace(layers.overrides, models=layers.overrides.models | {model_id: proposed})
    effective = _merge(layers, snapshot)
    await _keep_session_models(db, _merge(layers), effective)
    row = await db.get(LLMModelOverride, model_id)
    if row is None:
        row = LLMModelOverride(id=model_id)
        db.add(row)
    row.fields = entry_fields(proposed)
    row.disabled = proposed.disabled
    return await _changed(db, effective)


@router.put("/routing")
async def put_routing(
    body: RoutingPatch,
    db: Database,
    settings: Configuration,
    request: Request = _NO_REQUEST,
) -> dict[str, Any]:
    layers = await _begin_change(db, settings, request)
    active = _merge(layers).registry.models
    if any(getattr(body, name) is None for name in body.model_fields_set - {"light"}):
        raise HTTPException(
            422, "Назначение не может быть null. Используйте пустой список или объект."
        )
    names = [body.default] if body.default else []
    names += body.fallback or []
    names += list((body.roles or {}).values())
    if body.light:
        names.append(body.light)
    if any(name not in active for name in names) or (body.default == "" and active):
        raise HTTPException(422, "Для назначения выберите включённую модель.")
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
    request: Request | None = None,
    *,
    reset: bool = False,
) -> dict[str, Any]:
    layers = await _begin_change(db, settings, request)
    current = _merge(layers)
    item = next((row for row in current.view[category] if row["id"] == element_id), None)
    has_base = (item is not None and item["has_base"]) or element_id in current.view["deleted"][
        category
    ]
    if reset:
        if not has_base:
            raise HTTPException(404, "Для этой записи нет исходной настройки из файла.")
    elif item is None:
        raise HTTPException(404, "Настройка не найдена.")
    entries = dict(getattr(layers.overrides, category))
    if reset or (not disable and not has_base):
        entries.pop(element_id, None)
    elif disable:
        entries[element_id] = replace(entries.get(element_id, OverrideEntry()), disabled=True)
    else:
        entries[element_id] = OverrideEntry(deleted=True)
    routing = dict(layers.overrides.routing)
    removed_models: dict[str, OverrideEntry | None] = {}
    if not reset and not disable:
        affected = (
            {element_id}
            if category == "models"
            else {item["id"] for item in current.view["models"] if item["provider"] == element_id}
        )
        if routing.get("default") in affected:
            routing.pop("default")
        if routing.get("light") in affected:
            routing["light"] = None
        if "fallback" in routing:
            routing["fallback"] = [name for name in routing["fallback"] if name not in affected]
        if "roles" in routing:
            routing["roles"] = {
                role: name for role, name in routing["roles"].items() if name not in affected
            }
        if category == "providers" and not has_base:
            removed_models = {
                item["id"]: OverrideEntry(deleted=True) if item["has_base"] else None
                for item in current.view["models"]
                if item["id"] in affected
            }
    snapshot = replace(layers.overrides, **{category: entries}, routing=routing)
    if removed_models:
        models = dict(snapshot.models)
        for name, entry in removed_models.items():
            if entry is None:
                models.pop(name, None)
            else:
                models[name] = entry
        snapshot = replace(snapshot, models=models)
    effective = _merge(layers, snapshot)
    await _keep_session_models(db, current, effective)
    table = LLMProviderOverride if category == "providers" else LLMModelOverride
    row = cast(LLMProviderOverride | LLMModelOverride | None, await db.get(table, element_id))
    proposed = entries.get(element_id)
    if proposed is None:
        if row is not None:
            await db.delete(row)
    else:
        if row is None:
            row = table(id=element_id)
            db.add(row)
        row.fields, row.disabled = entry_fields(proposed), proposed.disabled
        if isinstance(row, LLMProviderOverride):
            row.api_key_ciphertext = proposed.api_key_ciphertext
    for name, entry in removed_models.items():
        model_row = await db.get(LLMModelOverride, name)
        if entry is None:
            if model_row is not None:
                await db.delete(model_row)
        else:
            if model_row is None:
                model_row = LLMModelOverride(id=name)
                db.add(model_row)
            model_row.fields, model_row.disabled = entry_fields(entry), entry.disabled
    routing_row = await db.get(LLMRoutingOverride, 1)
    assert routing_row is not None
    routing_row.fields = routing
    return await _changed(db, effective)


@router.delete("/providers/{provider_id:path}")
async def delete_provider(
    provider_id: ModelId,
    db: Database,
    settings: Configuration,
    disable: bool = False,
    request: Request = _NO_REQUEST,
) -> dict[str, Any]:
    return await _delete_element("providers", provider_id, disable, db, settings, request)


@router.delete("/models/{model_id:path}")
async def delete_model(
    model_id: ModelId,
    db: Database,
    settings: Configuration,
    disable: bool = False,
    request: Request = _NO_REQUEST,
) -> dict[str, Any]:
    return await _delete_element("models", model_id, disable, db, settings, request)


@router.post("/providers/{provider_id:path}/reset")
async def reset_provider(
    provider_id: ModelId,
    db: Database,
    settings: Configuration,
    request: Request = _NO_REQUEST,
) -> dict[str, Any]:
    return await _delete_element("providers", provider_id, False, db, settings, request, reset=True)


@router.post("/models/{model_id:path}/reset")
async def reset_model(
    model_id: ModelId, db: Database, settings: Configuration, request: Request = _NO_REQUEST
) -> dict[str, Any]:
    return await _delete_element("models", model_id, False, db, settings, request, reset=True)


# Listing fields that carry a limit; the first valid one wins.
_CONTEXT_FIELDS = ("max_context_length", "context_length", "context_window", "max_input_tokens")
_OUTPUT_FIELDS = ("max_output_tokens",)
# Listing fields that name the model type: "type" on a local server, "mode" on a gateway.
_KIND_FIELDS = ("type", "mode")


def _first_count(item: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = item.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _model_details(item: dict[str, Any]) -> dict[str, Any]:
    kind = next((item[name] for name in _KIND_FIELDS if isinstance(item.get(name), str)), None)
    state = item.get("state")
    return {
        "context_window": _first_count(item, _CONTEXT_FIELDS),
        "max_output_tokens": _first_count(item, _OUTPUT_FIELDS),
        "kind": kind,
        # A local server says whether the model is loaded into memory.
        "state": state if isinstance(state, str) else None,
    }


def _native_listing_url(base_url: str) -> str | None:
    """Return the native model listing of a local server, which gives the context length."""
    parts = urlsplit(base_url.rstrip("/"))
    if parts.hostname not in model_config._LOCAL_HOSTS or not parts.path.endswith("/v1"):
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path[:-3] + "/api/v0/models", "", ""))


def _empty_audit(destination: str, operation: str) -> OutboundAudit:
    return OutboundAudit(
        category="llm",
        destination=destination,
        operation=operation,
        payload_bytes=0,
        payload_sha256=hashlib.sha256(b"").hexdigest(),
        status="started",
    )


async def _native_details(
    db: AsyncSession, destination: str, model: model_config.ResolvedModel
) -> dict[str, dict[str, Any]]:
    """Read the context length and type of each model; a failure gives no details."""
    audit = _empty_audit(destination, "list_model_details")
    db.add(audit)
    await db.commit()
    details: dict[str, dict[str, Any]] = {}
    try:
        async with _transport(model) as (client, headers):
            response = await client.get(destination, headers=headers)
        audit.detail = str(response.status_code)
        if response.is_success:
            data = response.json()
            items = data.get("data") if isinstance(data, dict) else None
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    details[item["id"]] = _model_details(item)
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, UnicodeError):
        audit.detail = "Model details unavailable"
    audit.status = "completed" if details else "error"
    await db.commit()
    return details


def connection_client(*, verify: ssl.SSLContext | bool = True) -> httpx.AsyncClient:
    """Create an isolated client with a bounded timeout and no redirects."""
    return httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False, verify=verify)


@asynccontextmanager
async def _transport(
    model: model_config.ResolvedModel,
) -> AsyncIterator[tuple[httpx.AsyncClient, dict[str, str]]]:
    transport = LLMClient(model)
    try:
        headers, verify = await transport.transport_settings()
        client = connection_client() if verify is True else connection_client(verify=verify)
        async with client:
            yield client, headers
    finally:
        await transport.aclose()


def _connection_model(
    layers: ModelLayers, effective: EffectiveModels, provider_id: str, name: str
) -> model_config.ResolvedModel:
    provider = effective.config.providers[provider_id]
    return model_config.resolve_model(
        "connection",
        model_config.ModelSpec(provider=provider_id, model=name, context_window=4096, max_tokens=1),
        provider_id,
        provider,
        secret=effective.provider_keys[provider_id],
        configured=True,
        env=layers.env,
        warn=lambda _location, _message: None,
    )


@router.post("/providers/{provider_id:path}/test")
async def test_provider_connection(
    provider_id: ModelId, db: Database, settings: Configuration
) -> dict[str, Any]:
    layers = await _layers(db, settings)
    effective = _merge(layers)
    provider = effective.config.providers.get(provider_id)
    if provider is None:
        raise HTTPException(404, "Unknown or disabled provider")
    key = effective.provider_keys[provider_id]
    model = _connection_model(layers, effective, provider_id, "listing")
    if provider.api_key_env is not None and not key:
        return {"ok": False, "status": None, "models": [], "message": "API key is not configured"}
    destination = f"{provider.base_url.rstrip('/')}/models"
    audit = _empty_audit(destination, "list_models")
    db.add(audit)
    await db.commit()
    result: dict[str, Any] = {"ok": False, "status": None, "models": [], "details": []}
    listed: list[dict[str, Any]] = []
    try:
        async with _transport(model) as (client, headers):
            response = await client.get(destination, headers=headers)
        result["status"] = response.status_code
        if response.is_success:
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError("Invalid model listing")
            listed = [
                item
                for item in data["data"]
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and (not key or key not in item["id"])
            ]
            result["models"] = [item["id"] for item in listed]
            result["ok"] = True
        else:
            result["message"] = "Provider returned a non-success HTTP status"
    except (httpx.HTTPError, httpx.InvalidURL, LLMError):
        result["message"] = "Cannot connect to provider within the request timeout"
    except (ValueError, UnicodeError):
        result["message"] = "Provider returned an invalid model listing"
    audit.status = "completed" if result["ok"] else "error"
    audit.detail = str(result["status"]) if result["status"] is not None else result["message"]
    await db.commit()
    if result["ok"]:
        native = _native_listing_url(provider.base_url)
        extra = await _native_details(db, native, model) if native else {}
        result["details"] = [
            {"id": item["id"]}
            | _model_details(item)
            | {name: value for name, value in extra.get(item["id"], {}).items() if value}
            for item in listed
        ]
    return result


class ProbeRequest(_Patch):
    model: str = Field(min_length=1, max_length=200)


# A local server may load the model into memory first, so the check may take long.
PROBE_TIMEOUT_SECONDS = 120.0


async def _probe(db: AsyncSession, model: model_config.ResolvedModel) -> dict[str, Any]:
    destination = f"{model.base_url.rstrip('/')}/chat/completions"
    payload = build_wire_request(
        model,
        [{"role": "user", "content": "ok"}],
        [],
        max_tokens=1,
        reasoning_effort=None,
        stream=False,
    ).body
    audit = OutboundAudit(
        category="llm",
        destination=destination,
        operation="probe_model",
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        status="started",
    )
    db.add(audit)
    await db.commit()
    # The server's error text may name internal hosts, so only the status goes back.
    result: dict[str, Any] = {"ok": False, "status": None, "seconds": None}
    started = time.monotonic()
    try:
        async with _transport(model) as (client, headers):
            response = await client.post(
                destination,
                content=payload,
                headers=headers,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        result["status"] = response.status_code
        result["ok"] = response.is_success
        if not response.is_success:
            result["message"] = "The model did not answer"
    except httpx.TimeoutException:
        result["message"] = "The model did not answer within the time limit"
    except (httpx.HTTPError, httpx.InvalidURL, LLMError):
        result["message"] = "Cannot connect to provider"
    result["seconds"] = round(time.monotonic() - started, 2)
    audit.status = "completed" if result["ok"] else "error"
    audit.detail = str(result["status"]) if result["status"] is not None else result["message"]
    await db.commit()
    return result


@router.post("/providers/{provider_id:path}/probe")
async def probe_provider_model(
    provider_id: ModelId, body: ProbeRequest, db: Database, settings: Configuration
) -> dict[str, Any]:
    """Check a model that the server lists, before it is added."""
    layers = await _layers(db, settings)
    effective = _merge(layers)
    provider = effective.config.providers.get(provider_id)
    if provider is None:
        raise HTTPException(404, "Unknown or disabled provider")
    return await _probe(db, _connection_model(layers, effective, provider_id, body.model))


@router.post("/models/{model_id:path}/probe")
async def probe_agent_model(model_id: str, db: Database, settings: Configuration) -> dict[str, Any]:
    """Check an agent model; a model that does not answer is not offered for new sessions."""
    effective = _merge(await _layers(db, settings))
    model = effective.registry.models.get(model_id)
    if model is None:
        raise HTTPException(404, "Unknown or disabled model")
    result = await _probe(db, model)
    model_health.record(
        model_id,
        model.base_url,
        model.model,
        result["ok"],
        result["status"],
        effective.registry.checksum,
    )
    return result


# Default ports of common local model servers.
LOCAL_SERVER_PORTS = (1234, 11434, 8080)
SCAN_TIMEOUT_SECONDS = 1.5


def local_port_files() -> list[Path]:
    """Files where a local desktop server keeps its own port, if it is installed."""
    return [Path.home() / ".lmstudio" / ".internal" / "http-server-config.json"]


def _local_ports() -> list[int]:
    ports = list(LOCAL_SERVER_PORTS)
    for path in local_port_files():
        try:
            port = json.loads(path.read_text(encoding="utf-8")).get("port")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(port, int) and 0 < port < 65536 and port not in ports:
            ports.insert(0, port)
    return ports


async def _answers(client: httpx.AsyncClient, url: str) -> Any:
    try:
        response = await client.get(url, timeout=SCAN_TIMEOUT_SECONDS)
        return response.json() if response.is_success else None
    except (httpx.HTTPError, ValueError, UnicodeError):
        return None


@router.post("/local-servers")
async def scan_local_servers(db: Database, settings: Configuration) -> list[dict[str, Any]]:
    """Find model servers that run on this computer now, without any setting."""
    effective = _merge(await _layers(db, settings))
    connected = {item["base_url"].rstrip("/"): item["id"] for item in effective.view["providers"]}
    audit = _empty_audit("http://127.0.0.1", "scan_local_servers")
    db.add(audit)
    await db.commit()
    found: list[dict[str, Any]] = []
    async with connection_client() as client:
        for port in _local_ports():
            root = f"http://127.0.0.1:{port}"
            listing = await _answers(client, f"{root}/v1/models")
            items = listing.get("data") if isinstance(listing, dict) else None
            if not isinstance(items, list):
                continue
            if await _answers(client, f"{root}/api/v0/models") is not None:
                server = "lmstudio"
            elif await _answers(client, f"{root}/api/version") is not None:
                server = "ollama"
            else:
                server = "openai"
            base_url = f"{root}/v1"
            found.append(
                {
                    "base_url": base_url,
                    "server": server,
                    "models": sum(isinstance(item, dict) for item in items),
                    "provider": connected.get(base_url),
                }
            )
    audit.status = "completed"
    audit.detail = f"{len(found)} found"
    await db.commit()
    return found
