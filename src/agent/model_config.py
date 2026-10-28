"""Validate model declarations and resolve credentials without runtime side effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, Protocol, Self
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails
from yaml.error import MarkedYAMLError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from agent.config import EngineProfileSettings

# openai: OpenAI-compatible /chat/completions (OpenAI, GLM, LM Studio, MLX, Ollama, gateways).
# openai_responses: OpenAI Responses API (Codex/GPT reasoning models). anthropic: Messages API.
ProviderKind = Literal["openai", "openai_responses", "anthropic", "jev"]
# Transports the runtime can drive today; the rest are declared ahead of their adapters.
SUPPORTED_TRANSPORTS: frozenset[str] = frozenset({"openai"})
FieldPath = tuple[str | int, ...]
ModelId = Annotated[
    # ":", "/" and "@" keep legacy profile IDs such as "local:qwen" valid.
    str, StringConstraints(min_length=1, max_length=100, pattern=r"\A[A-Za-z0-9._:/@-]+\z")
]
_MAX_FILE_BYTES = 1024 * 1024
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})
# Environment variable names and ${NAME} references in header values and model templates.
EnvName = Annotated[str, StringConstraints(pattern=r"\A[A-Za-z_][A-Za-z0-9_]{0,127}\z")]
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]{0,127})\}")
_HEADER_NAME = re.compile(r"\A[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}\Z")
_YAML_TAGS = frozenset(
    f"tag:yaml.org,2002:{name}" for name in ("map", "seq", "str", "null", "bool", "int", "float")
)


class ModelConfigError(ValueError):
    """Report configuration errors without input values."""


class _Spec(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )


def _check_http_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("must be a valid HTTP(S) URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or port == 0
        or any(character.isspace() or ord(character) < 32 for character in value)
        or "\\" in value
    ):
        raise ValueError("must be a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None or "?" in value or "#" in value:
        raise ValueError("URL must not contain userinfo, query or fragment")
    return value


class HeaderKeyAuthSpec(_Spec):
    """A key in a header, for example "Authorization: Api-Key <key>"."""

    type: Literal["header_key"]
    key_env: EnvName
    header: str = "Authorization"
    prefix: str = "Api-Key "


class BearerAuthSpec(_Spec):
    """A static token in "Authorization: Bearer <token>"."""

    type: Literal["bearer"]
    key_env: EnvName


class OAuthAuthSpec(_Spec):
    """A key that the provider exchanges for a short-lived token (OAuth client credentials)."""

    type: Literal["oauth_client_credentials"]
    token_url: str = Field(repr=False)
    key_env: EnvName
    scope: str = Field(min_length=1, max_length=200)
    request_id_header: str | None = "RqUID"
    expires_at_unit: Literal["ms", "s"] = "ms"

    @field_validator("token_url")
    @classmethod
    def validate_token_url(cls, value: str) -> str:
        _check_http_url(value)
        if urlsplit(value).scheme != "https":
            raise ValueError("token URL must use HTTPS")
        return value


class CommandAuthSpec(_Spec):
    """A token printed by a local command, without a shell."""

    type: Literal["command"]
    argv: list[str] = Field(min_length=1, max_length=32)
    ttl_seconds: float = Field(default=300, gt=0, le=86400)


class NoAuthSpec(_Spec):
    type: Literal["none"]


AuthSpec = Annotated[
    HeaderKeyAuthSpec | BearerAuthSpec | OAuthAuthSpec | CommandAuthSpec | NoAuthSpec,
    Field(discriminator="type"),
]


class ProviderSpec(_Spec):
    kind: ProviderKind
    base_url: str = Field(repr=False)
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    # Flexible provider features.
    auth: AuthSpec | None = None
    # Extra headers; a value may reference the environment as ${NAME}.
    headers: dict[str, str] = Field(default_factory=dict, max_length=32)
    ca_bundle: str | None = None
    dialect: Literal["chat", "functions"] = "chat"
    # Provider model ID built from the model name, for example "${FOLDER}/{model}/latest".
    model_template: str | None = Field(default=None, max_length=300)
    # Data stays in Russia: projects with personal data may use only such providers.
    data_in_russia: bool = False

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        for name, header_value in value.items():
            if not _HEADER_NAME.fullmatch(name):
                raise ValueError("contains an invalid header name")
            if any(ord(character) < 32 for character in header_value):
                raise ValueError("contains a control character in a header value")
        return value

    @field_validator("model_template")
    @classmethod
    def validate_template(cls, value: str | None) -> str | None:
        if value is not None and value.count("{model}") != 1:
            raise ValueError("must contain {model} exactly once")
        return value

    @model_validator(mode="after")
    def validate_auth(self) -> Self:
        if self.auth is not None and self.api_key_env is not None:
            raise ValueError("use either api_key_env or auth, not both")
        return self

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _check_http_url(value)


class ReasoningTagsSpec(_Spec):
    """Tags around a reasoning block that a model writes into its answer text."""

    start: str = Field(min_length=1)
    end: str = Field(min_length=1)


class ReasoningSpec(_Spec):
    supported: bool = False
    allowed_efforts: list[str] = Field(default_factory=list)
    default_effort: str | None = None
    wire_parameter: Literal["reasoning_effort", "thinking_budget", "enable_thinking"] | None = None
    # Which earlier reasoning texts go back to the model: none, the current turn (since the
    # last user message, for tool loops) or all of them.
    history: Literal["none", "current_turn", "all"] = "none"
    # The block between these tags moves from the answer to the reasoning; see reasoning_tags.
    tags: ReasoningTagsSpec | None = None

    @field_validator("allowed_efforts")
    @classmethod
    def validate_efforts(cls, value: list[str], info: ValidationInfo) -> list[str]:
        if not set(value) <= _EFFORTS:
            raise ValueError("contains unsupported reasoning efforts")
        if value and not info.data.get("supported", False):
            raise ValueError("must be empty when reasoning is not supported")
        return value

    @field_validator("default_effort")
    @classmethod
    def validate_default(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is not None and value not in info.data.get("allowed_efforts", []):
            raise ValueError("must belong to allowed_efforts")
        return value


class PricingSpec(_Spec):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class ModelSpec(_Spec):
    provider: str
    model: str = Field(min_length=1)
    context_window: int = Field(ge=4096)
    max_tokens: int = Field(gt=0)
    reasoning: ReasoningSpec = Field(default_factory=ReasoningSpec)
    pricing: PricingSpec | None = None
    inference_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False, strict=True)
    semantic_idle_timeout_seconds: float = Field(
        default=120, gt=0, allow_inf_nan=False, strict=True
    )
    repetition_threshold_chars: int | None = Field(default=65536, gt=0, strict=True)

    @field_validator("inference_timeout_seconds", "semantic_idle_timeout_seconds", mode="before")
    @classmethod
    def validate_guard_number(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a finite positive number")
        return value

    @model_validator(mode="after")
    def validate_guard_limits(self) -> Self:
        if self.semantic_idle_timeout_seconds > self.inference_timeout_seconds:
            raise ValueError(
                "semantic_idle_timeout_seconds must not exceed inference_timeout_seconds"
            )
        return self

    @field_validator("max_tokens")
    @classmethod
    def validate_tokens(cls, value: int, info: ValidationInfo) -> int:
        context_window = info.data.get("context_window")
        if context_window is not None and value >= context_window:
            raise ValueError("must be smaller than context_window")
        return value


class RoutingSpec(_Spec):
    default: str
    fallback: list[str] = Field(default_factory=list)
    roles: dict[str, str] = Field(default_factory=dict)
    # Model for internal tasks: context summaries and session titles. None: use the default.
    light: str | None = None


class DecisionsSpec(_Spec):
    enabled: bool = False
    provider: str | None = None
    model: str | None = None

    @field_validator("enabled")
    @classmethod
    def validate_enabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("not supported yet")
        return value


def _validation_issue(location: FieldPath, message: str) -> InitErrorDetails:
    return {
        "type": "value_error",
        "loc": location,
        "input": None,
        "ctx": {"error": ValueError(message)},
    }


class ModelsConfig(_Spec):
    providers: dict[str, ProviderSpec] = Field(min_length=1)
    models: dict[ModelId, ModelSpec] = Field(min_length=1)
    routing: RoutingSpec
    decisions: DecisionsSpec = Field(default_factory=DecisionsSpec)
    engine_profiles: dict[ModelId, EngineProfileSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        errors: list[InitErrorDetails] = []
        for model_id, model in self.models.items():
            if model.provider not in self.providers:
                errors.append(
                    _validation_issue(
                        ("models", model_id, "provider"), "unknown provider reference"
                    )
                )
        if self.decisions.provider is not None and self.decisions.provider not in self.providers:
            errors.append(
                _validation_issue(("decisions", "provider"), "unknown provider reference")
            )
        selections: list[tuple[FieldPath, str]] = [(("routing", "default"), self.routing.default)]
        selections.extend(
            (("routing", "fallback", index), model_id)
            for index, model_id in enumerate(self.routing.fallback)
        )
        selections.extend(
            (("routing", "roles", role), model_id) for role, model_id in self.routing.roles.items()
        )
        if self.routing.light is not None:
            selections.append((("routing", "light"), self.routing.light))
        for location, model_id in selections:
            selected_model = self.models.get(model_id)
            if selected_model is None:
                errors.append(_validation_issue(location, "unknown model reference"))
            elif (
                selected_model.provider in self.providers
                and self.providers[selected_model.provider].kind not in SUPPORTED_TRANSPORTS
            ):
                errors.append(_validation_issue(location, "transport is not supported yet"))
        if errors:
            raise ValidationError.from_exception_data(type(self).__name__, errors, hide_input=True)
        return self


class LegacyLLMProfile(Protocol):
    """Expose the existing profile settings structurally."""

    @property
    def base_url(self) -> str: ...
    @property
    def api_key(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def timeout_seconds(self) -> float: ...
    @property
    def max_retries(self) -> int: ...
    @property
    def max_tokens(self) -> int: ...


class LegacyLLMSettings(Protocol):
    """Accept Settings without importing the application configuration."""

    @property
    def llm_base_url(self) -> str: ...
    @property
    def llm_api_key(self) -> str: ...
    @property
    def llm_model(self) -> str: ...
    @property
    def llm_timeout_seconds(self) -> float: ...
    @property
    def llm_max_retries(self) -> int: ...
    @property
    def llm_max_tokens(self) -> int: ...
    @property
    def llm_profiles(self) -> Mapping[str, LegacyLLMProfile]: ...
    @property
    def default_llm_profile(self) -> str: ...
    @property
    def context_window(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    id: str
    provider_id: str
    kind: ProviderKind
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    configured: bool
    model: str
    context_window: int
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    reasoning: ReasoningSpec
    pricing: PricingSpec | None
    # Flexible provider features; the defaults describe a plain Bearer-key provider.
    auth: AuthSpec | None = None
    auth_secret: str = field(default="", repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    ca_bundle: str | None = None
    dialect: Literal["chat", "functions"] = "chat"
    data_in_russia: bool = False
    inference_timeout_seconds: float = 600
    semantic_idle_timeout_seconds: float = 120
    repetition_threshold_chars: int | None = 65536


@dataclass(frozen=True, slots=True)
class ResolvedModelRegistry:
    models: dict[str, ResolvedModel]
    default_id: str
    fallback: list[str]
    roles: dict[str, str]
    warnings: list[str]
    source: Literal["yaml", "legacy"]
    checksum: str
    light_id: str | None = None

    def light(self) -> ResolvedModel:
        """Return the model for internal tasks, or the default model when none is set."""
        return self.get(self.light_id or self.default_id)

    def get(self, model_id: str) -> ResolvedModel:
        """Return a model or explain an unknown ID."""
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(f"Unknown model ID: {model_id!r}") from None

    def ids(self) -> list[str]:
        """Return model IDs in declaration order."""
        return list(self.models)


@dataclass(slots=True)
class _Diagnostics:
    filename: str
    locations: dict[FieldPath, tuple[int, int]] = field(default_factory=dict)

    def message(self, location: FieldPath, message: str) -> str:
        field_path = ""
        for part in location:
            field_path += f"[{part}]" if isinstance(part, int) else f".{part}"
        nearest = location
        while nearest and nearest not in self.locations:
            nearest = nearest[:-1]
        line, column = self.locations.get(nearest, (1, 1))
        return f"{self.filename}:{line}:{column}: {field_path.lstrip('.') or '<root>'}: {message}"

    def fail(self, location: FieldPath, message: str) -> None:
        raise ModelConfigError(self.message(location, message))


def _inspect_yaml(
    node: Node, diagnostics: _Diagnostics, location: FieldPath, active: set[int], visits: list[int]
) -> None:
    diagnostics.locations[location] = (node.start_mark.line + 1, node.start_mark.column + 1)
    visits[0] += 1
    if len(location) > 100 or visits[0] > 20_000:
        diagnostics.fail(location, "YAML nesting or alias expansion limit exceeded")
    if id(node) in active:
        diagnostics.fail(location, "recursive YAML aliases are not allowed")
    if node.tag not in _YAML_TAGS:
        diagnostics.fail(location, "unsupported YAML tag")
    active.add(id(node))
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                diagnostics.fail(location, "mapping keys must be strings; merges are not allowed")
            key: str = key_node.value
            child = (*location, key)
            if key in seen:
                diagnostics.locations[child] = (
                    key_node.start_mark.line + 1,
                    key_node.start_mark.column + 1,
                )
                diagnostics.fail(child, "duplicate mapping key")
            seen.add(key)
            _inspect_yaml(value_node, diagnostics, child, active, visits)
    elif isinstance(node, SequenceNode):
        for index, child_node in enumerate(node.value):
            _inspect_yaml(child_node, diagnostics, (*location, index), active, visits)
    active.remove(id(node))


def _parse_yaml(content: bytes, diagnostics: _Diagnostics) -> object:
    loader: yaml.SafeLoader | None = None
    try:
        loader = yaml.SafeLoader(content.decode("utf-8"))
        node = loader.get_single_node()
        if node is None:
            diagnostics.fail((), "configuration is empty")
            return None
        _inspect_yaml(node, diagnostics, (), set(), [0])
        return loader.construct_object(node, deep=True)
    except ModelConfigError:
        raise
    except MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        if mark is not None:
            diagnostics.locations[()] = (mark.line + 1, mark.column + 1)
        raise ModelConfigError(diagnostics.message((), "invalid YAML syntax")) from None
    except (yaml.YAMLError, UnicodeError, ValueError, RecursionError):
        raise ModelConfigError(diagnostics.message((), "invalid YAML document")) from None
    finally:
        if loader is not None:
            loader.dispose()


def _validate(data: object, diagnostics: _Diagnostics) -> ModelsConfig:
    try:
        return ModelsConfig.model_validate(data)
    except ValidationError as exc:
        messages = [
            diagnostics.message(
                tuple(part for part in error["loc"] if part != "[key]"), error["msg"]
            )
            for error in exc.errors(include_url=False, include_input=False, include_context=False)
        ]
        raise ModelConfigError("\n".join(messages)) from None


def _expand(value: str, env: Mapping[str, str], missing: set[str]) -> str:
    """Replace each ${NAME} with its environment value; collect the names that are not set."""

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if not env.get(name):
            missing.add(name)
            return ""
        return env[name]

    return _ENV_REFERENCE.sub(substitute, value)


def secret_env(provider: ProviderSpec) -> str | None:
    """Return the environment variable that holds the provider secret, if any."""
    if provider.auth is not None:
        return getattr(provider.auth, "key_env", None)
    return provider.api_key_env


def resolve_model(
    model_id: str,
    model: ModelSpec,
    provider_id: str,
    provider: ProviderSpec,
    *,
    secret: str,
    configured: bool,
    env: Mapping[str, str],
    warn: Callable[[FieldPath, str], None],
) -> ResolvedModel:
    """Build one runtime model; the YAML loader and the UI override layer share it.

    ``configured`` is the caller's key decision. Missing ${NAME} values make the model
    unconfigured with a warning.
    """
    missing: set[str] = set()
    headers = {name: _expand(value, env, missing) for name, value in provider.headers.items()}
    name = model.model
    if provider.model_template is not None:
        name = _expand(provider.model_template, env, missing).replace("{model}", model.model)
    if missing:
        warn(
            ("models", model_id),
            "environment variables for headers or model_template are not set: "
            + ", ".join(sorted(missing)),
        )
        configured = False
    return ResolvedModel(
        id=model_id,
        provider_id=provider_id,
        kind=provider.kind,
        base_url=provider.base_url,
        # The legacy Bearer key only; an auth block keeps its secret apart.
        api_key=secret if provider.auth is None else "",
        configured=configured,
        model=name,
        context_window=model.context_window,
        max_tokens=model.max_tokens,
        timeout_seconds=provider.timeout_seconds,
        max_retries=provider.max_retries,
        reasoning=model.reasoning,
        pricing=model.pricing,
        inference_timeout_seconds=model.inference_timeout_seconds,
        semantic_idle_timeout_seconds=model.semantic_idle_timeout_seconds,
        repetition_threshold_chars=model.repetition_threshold_chars,
        auth=provider.auth,
        auth_secret=secret if provider.auth is not None else "",
        headers=headers,
        ca_bundle=provider.ca_bundle,
        dialect=provider.dialect,
        data_in_russia=provider.data_in_russia,
    )


def _resolve(
    config: ModelsConfig,
    credentials: Mapping[str, str],
    diagnostics: _Diagnostics,
    source: Literal["yaml", "legacy"],
    checksum: str,
    warnings: list[str],
    env: Mapping[str, str] | None = None,
) -> ResolvedModelRegistry:
    env = env or {}
    for provider_id, provider in config.providers.items():
        parsed = urlsplit(provider.base_url)
        if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
            warnings.append(
                diagnostics.message(
                    ("providers", provider_id, "base_url"),
                    "non-local HTTP base URL is not encrypted",
                )
            )

    def warn(location: FieldPath, message: str) -> None:
        text = diagnostics.message(location, message)
        if text not in warnings:
            warnings.append(text)

    models: dict[str, ResolvedModel] = {}
    for model_id, model in config.models.items():
        provider = config.providers[model.provider]
        secret = credentials[model.provider]
        key_env = secret_env(provider)
        if provider.auth is None:
            configured = bool(secret) or (source == "yaml" and key_env is None)
        else:
            configured = bool(secret) or key_env is None
        models[model_id] = resolve_model(
            model_id,
            model,
            model.provider,
            provider,
            secret=secret,
            configured=configured,
            env=env,
            warn=warn,
        )
    return ResolvedModelRegistry(
        models=models,
        default_id=config.routing.default,
        fallback=list(config.routing.fallback),
        roles=dict(config.routing.roles),
        warnings=warnings,
        source=source,
        light_id=config.routing.light,
        checksum=checksum,
    )


def _load_legacy(legacy: LegacyLLMSettings | None) -> ResolvedModelRegistry:
    diagnostics = _Diagnostics("<legacy>")
    if legacy is None:
        raise ModelConfigError(diagnostics.message((), "no YAML file or legacy settings provided"))
    providers: dict[str, object] = {}
    models: dict[str, object] = {}
    credentials: dict[str, str] = {}
    if legacy.llm_profiles:
        profiles = legacy.llm_profiles
    else:
        profiles = {
            "default": SimpleNamespace(
                base_url=legacy.llm_base_url,
                api_key=legacy.llm_api_key,
                model=legacy.llm_model,
                timeout_seconds=legacy.llm_timeout_seconds,
                max_retries=legacy.llm_max_retries,
                max_tokens=legacy.llm_max_tokens,
            )
        }
    for name, profile in profiles.items():
        providers[name] = {
            "kind": "openai",
            "base_url": profile.base_url,
            "timeout_seconds": profile.timeout_seconds,
            "max_retries": profile.max_retries,
        }
        models[name] = {
            "provider": name,
            "model": profile.model,
            "context_window": legacy.context_window,
            "max_tokens": profile.max_tokens,
        }
        credentials[name] = profile.api_key
    config = _validate(
        {
            "providers": providers,
            "models": models,
            "routing": {"default": legacy.default_llm_profile},
        },
        diagnostics,
    )
    checksum = hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest()
    warnings = (
        ["AGENT_LLM_PROFILES is deprecated; migrate to config/models.yaml"]
        if legacy.llm_profiles
        else []
    )
    return _resolve(config, credentials, diagnostics, "legacy", checksum, warnings)


def load_models(
    models_file: Path | None, env: Mapping[str, str], legacy: LegacyLLMSettings | None
) -> ResolvedModelRegistry:
    """Load YAML when present, otherwise preserve legacy profile selection."""
    if models_file is None:
        return _load_legacy(legacy)
    diagnostics = _Diagnostics(str(models_file))
    try:
        with models_file.open("rb") as stream:
            content = stream.read(_MAX_FILE_BYTES + 1)
    except FileNotFoundError:
        return _load_legacy(legacy)
    except OSError:
        raise ModelConfigError(diagnostics.message((), "cannot read configuration file")) from None
    if len(content) > _MAX_FILE_BYTES:
        raise ModelConfigError(diagnostics.message((), "configuration exceeds the 1 MiB limit"))
    config = _validate(_parse_yaml(content, diagnostics), diagnostics)
    override = env.get("AGENT_DEFAULT_LLM_PROFILE")
    override_warnings: list[str] = []
    if override is not None and override != config.routing.default:
        if override in config.models:
            data = config.model_dump()
            data["routing"]["default"] = override
            config = _validate(data, diagnostics)
        else:
            # Typically a leftover legacy value such as "default"; YAML routing stays in charge.
            override_warnings.append(
                "AGENT_DEFAULT_LLM_PROFILE names no model in the YAML file and is ignored"
            )
    credentials: dict[str, str] = {}
    warnings_missing: list[str] = []
    for provider_id, provider in config.providers.items():
        key_env = secret_env(provider)
        api_key = env.get(key_env, "") if key_env is not None else ""
        if key_env is not None and not api_key:
            # A declared-but-unkeyed provider must not block startup; it stays unconfigured.
            location: FieldPath = (
                ("providers", provider_id, "api_key_env")
                if provider.auth is None
                else ("providers", provider_id, "auth", "key_env")
            )
            warnings_missing.append(
                diagnostics.message(
                    location, "API key environment variable is not set; provider is unconfigured"
                )
            )
        credentials[provider_id] = api_key
    structural_vars = {
        "AGENT_LLM_PROFILES",
        "AGENT_LLM_BASE_URL",
        "AGENT_LLM_MODEL",
        "AGENT_LLM_TIMEOUT_SECONDS",
        "AGENT_LLM_MAX_RETRIES",
        "AGENT_LLM_MAX_TOKENS",
    }
    warnings = (
        override_warnings
        + warnings_missing
        + (
            ["Legacy LLM structure variables are ignored when YAML is present"]
            if structural_vars.intersection(env)
            else []
        )
    )
    return _resolve(
        config,
        credentials,
        diagnostics,
        "yaml",
        hashlib.sha256(content).hexdigest(),
        warnings,
        env,
    )


def check_main(argv: list[str] | None = None) -> int:
    """Check configuration without initializing directories, API or database."""
    parser = argparse.ArgumentParser(prog="agent-config-check")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    from dotenv import dotenv_values

    from agent.config import PROJECT_ROOT, Settings, _env_file

    try:
        env_file = _env_file()
        env = (
            {name: value for name, value in dotenv_values(env_file).items() if value is not None}
            if env_file is not None
            else {}
        )
        env.update(os.environ)
        explicit_file = args.file is not None or "AGENT_MODELS_FILE" in env
        models_file = args.file or Path(
            env.get("AGENT_MODELS_FILE", str(PROJECT_ROOT / "config" / "models.yaml"))
        )
        if explicit_file and not models_file.is_file():
            raise ModelConfigError(
                _Diagnostics(str(models_file)).message((), "configuration file is missing")
            )
        legacy = None if models_file.exists() else Settings()
        registry = load_models(models_file, env, legacy)
    except ValidationError as exc:
        fields = ", ".join(
            ".".join(map(str, error["loc"]))
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        )
        print(f"ERROR: <legacy>:1:1: {fields}: invalid legacy settings", file=sys.stderr)
        return 1
    except ModelConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError):
        print("ERROR: <environment>:1:1: <root>: cannot read environment file", file=sys.stderr)
        return 1
    for warning in registry.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.strict and registry.warnings:
        print("ERROR: warnings are forbidden by --strict", file=sys.stderr)
        return 2
    print(
        f"OK: {registry.source}, {len(registry.models)} models, default={registry.default_id}, "
        f"checksum={registry.checksum}"
    )
    return 0
