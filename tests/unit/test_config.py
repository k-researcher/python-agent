import pytest

from agent.config import Settings


def test_external_bind_requires_api_token() -> None:
    with pytest.raises(RuntimeError, match="AGENT_API_TOKEN"):
        Settings(host="0.0.0.0", api_token="").validate_runtime_security()


def test_local_bind_does_not_require_api_token() -> None:
    Settings(host="127.0.0.1", api_token="").validate_runtime_security()


def test_distributed_mode_requires_postgresql() -> None:
    with pytest.raises(RuntimeError, match="requires PostgreSQL"):
        Settings(
            execution_mode="redis",
            database_url="sqlite+aiosqlite:///agent.db",
        ).validate_runtime_security()


def test_distributed_mode_accepts_async_postgresql() -> None:
    Settings(
        execution_mode="redis",
        database_url="postgresql+asyncpg://agent:secret@db/agent",
    ).validate_runtime_security()


def test_named_llm_profiles_resolve_independently() -> None:
    settings = Settings(
        default_llm_profile="fast",
        llm_profiles={
            "fast": {
                "base_url": "https://fast.example/v1",
                "api_key": "fast-key",
                "model": "fast-model",
            },
            "reasoning": {
                "base_url": "https://reasoning.example/v1",
                "api_key": "reasoning-key",
                "model": "reasoning-model",
                "max_tokens": 16_000,
            },
        },
    )

    settings.validate_runtime_security()
    assert settings.resolve_llm_profile().model == "fast-model"
    assert settings.resolve_llm_profile("reasoning").max_tokens == 16_000


def test_unknown_llm_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown LLM profile"):
        Settings().resolve_llm_profile("missing")


def test_llm_profile_requires_http_url() -> None:
    settings = Settings(
        llm_profiles={
            "invalid": {"base_url": "file:///tmp/model", "model": "unsafe-model"}
        },
        default_llm_profile="invalid",
    )
    with pytest.raises(RuntimeError, match=r"HTTP\(S\)"):
        settings.validate_runtime_security()
