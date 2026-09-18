from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProfileSettings(BaseModel):
    base_url: str = Field(min_length=1)
    api_key: str = ""
    model: str = Field(min_length=1)
    timeout_seconds: float = 120.0
    max_retries: int = Field(default=2, ge=0, le=10)
    max_tokens: int = Field(default=8192, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
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

    data_dir: Path = PROJECT_ROOT / "data"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate_runtime_security(self) -> None:
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.api_token:
            raise RuntimeError("AGENT_API_TOKEN is required when binding outside localhost")
        if self.execution_mode == "redis" and not self.database_url.startswith(
            ("postgresql+asyncpg://", "postgres+asyncpg://")
        ):
            raise RuntimeError(
                "Redis execution mode requires PostgreSQL via postgresql+asyncpg://"
            )
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
