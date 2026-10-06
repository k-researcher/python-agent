from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def normalize_origin(value: str) -> str:
    """Canonical scheme://host[:port] form; default ports are dropped."""
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Invalid origin: {value!r}")
    if parts.username or parts.password or parts.path not in {"", "/"} or parts.query:
        raise ValueError(f"Origin must not contain credentials, path or query: {value!r}")
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if parts.scheme == "https" else 80
    port = f":{parts.port}" if parts.port and parts.port != default_port else ""
    return f"{parts.scheme}://{host}{port}"


def _env_file() -> Path | None:
    """AGENT_ENV_FILE overrides the dotenv path; an empty value disables it (used by tests)."""
    override = os.environ.get("AGENT_ENV_FILE")
    if override is None:
        return PROJECT_ROOT / ".env"
    return Path(override) if override else None


class LLMProfileSettings(BaseModel):
    base_url: str = Field(min_length=1)
    api_key: str = ""
    model: str = Field(min_length=1)
    timeout_seconds: float = 120.0
    max_retries: int = Field(default=2, ge=0, le=10)
    max_tokens: int = Field(default=8192, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file(),
        env_prefix="AGENT_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8080
    database_url: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'data' / 'agent.db'}"
    execution_mode: Literal["embedded", "redis"] = "embedded"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_namespace: str = "python-agent"
    worker_name: str = ""
    worker_lock_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)
    worker_visibility_timeout_seconds: int = Field(default=300, ge=30, le=3600)

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 8192
    default_llm_profile: str = Field(default="default", min_length=1, max_length=100)
    llm_profiles: dict[str, LLMProfileSettings] = Field(default_factory=dict)
    context_window: int = Field(default=128_000, ge=4096)
    context_reserved_tokens: int = Field(default=12_000, ge=1024)
    max_inference_steps: int = Field(default=100, ge=1, le=1000)
    max_tool_result_chars: int = Field(default=30_000, ge=1000, le=1_000_000)

    raw_llm_log: bool = False
    knowledge_base_enabled: bool = False
    allow_network_tools: bool = False
    network_allowlist: list[str] = Field(default_factory=list)
    http_max_response_bytes: int = Field(default=1_000_000, ge=1024, le=20_000_000)
    web_search_url_template: str = ""
    database_connections: dict[str, str] = Field(default_factory=dict)
    ssh_connections: dict[str, dict[str, Any]] = Field(default_factory=dict)
    knowledge_base_embedding_url: str = ""
    knowledge_base_embedding_api_key: str = ""
    knowledge_base_embedding_model: str = ""
    max_parallel_sessions: int = Field(default=4, ge=1, le=32)
    max_child_depth: int = Field(default=3, ge=0, le=10)
    max_child_wait_seconds: int = Field(default=600, ge=10, le=7200)
    api_token: str = ""

    # Browser access: exact Host names, the public origin of the UI and auth lifetimes.
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "[::1]"]
    )
    public_origin: str = ""
    extra_origins: list[str] = Field(default_factory=list)
    trusted_proxy_ips: list[str] = Field(default_factory=list)
    auth_link_ttl_seconds: int = Field(default=600, ge=60, le=86_400)
    auth_session_ttl_seconds: int = Field(default=30 * 86_400, ge=3600, le=365 * 86_400)
    auth_open_browser: bool = True

    # Model catalogue: unset = config/models.yaml if present; "" = legacy AGENT_LLM_*;
    # a path = that file, which must exist.
    models_file: str | None = None

    data_dir: Path = PROJECT_ROOT / "data"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def binds_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def effective_public_origin(self) -> str:
        if self.public_origin:
            return normalize_origin(self.public_origin)
        return f"http://127.0.0.1:{self.port}"

    def allowed_origins(self) -> frozenset[str]:
        origins = {normalize_origin(item) for item in self.extra_origins}
        if self.public_origin:
            origins.add(normalize_origin(self.public_origin))
        else:
            origins.update(
                f"http://{host}:{self.port}" for host in ("127.0.0.1", "localhost", "[::1]")
            )
        return frozenset(origins)

    @property
    def secure_cookies(self) -> bool:
        return self.effective_public_origin().startswith("https://")

    def validate_runtime_security(self) -> None:
        if not self.binds_loopback and not self.public_origin:
            raise RuntimeError(
                "AGENT_PUBLIC_ORIGIN is required when binding outside localhost"
            )
        if self.public_origin:
            origin = normalize_origin(self.public_origin)
            hostname = urlsplit(origin).hostname or ""
            if hostname not in {host.strip("[]") for host in self.allowed_hosts}:
                raise RuntimeError("AGENT_PUBLIC_ORIGIN host must be listed in AGENT_ALLOWED_HOSTS")
            # A container binds 0.0.0.0 but may publish only on 127.0.0.1; HTTP is then local.
            if not origin.startswith("https://") and hostname not in LOOPBACK_HOSTS:
                raise RuntimeError("AGENT_PUBLIC_ORIGIN must use https:// for remote access")
        if any("*" in host for host in self.allowed_hosts):
            raise RuntimeError("AGENT_ALLOWED_HOSTS must list exact host names")
        if self.execution_mode == "redis" and not self.database_url.startswith(
            ("postgresql+asyncpg://", "postgres+asyncpg://")
        ):
            raise RuntimeError(
                "Redis execution mode requires PostgreSQL via postgresql+asyncpg://"
            )
        if self.models_file is None and (PROJECT_ROOT / "config" / "models.yaml").is_file():
            return  # agent.model_registry validates the YAML file.
        if self.models_file is not None and str(self.models_file).strip():
            if not Path(self.models_file).expanduser().is_file():
                raise RuntimeError(f"AGENT_MODELS_FILE does not exist: {self.models_file}")
            return
        self.resolve_llm_profile(self.default_llm_profile)
        for name, profile in self.available_llm_profiles().items():
            if not 1 <= len(name) <= 100:
                raise RuntimeError("LLM profile names must contain 1-100 characters")
            parsed = urlsplit(profile.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise RuntimeError(f"LLM profile {name} must use an HTTP(S) base URL")

    def resolve_llm_profile(self, name: str | None = None) -> LLMProfileSettings:
        profile_name = name or self.default_llm_profile
        if self.llm_profiles:
            profile = self.llm_profiles.get(profile_name)
            if profile is None:
                raise ValueError(f"Unknown LLM profile: {profile_name}")
            return profile
        if profile_name != "default":
            raise ValueError(f"Unknown LLM profile: {profile_name}")
        return LLMProfileSettings(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
            timeout_seconds=self.llm_timeout_seconds,
            max_retries=self.llm_max_retries,
            max_tokens=self.llm_max_tokens,
        )

    def available_llm_profiles(self) -> dict[str, LLMProfileSettings]:
        if self.llm_profiles:
            return self.llm_profiles
        return {"default": self.resolve_llm_profile("default")}


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_directories()
    return settings
