import pytest

from agent.config import Settings


def test_external_bind_requires_public_origin() -> None:
    with pytest.raises(RuntimeError, match="AGENT_PUBLIC_ORIGIN"):
        Settings(host="0.0.0.0", public_origin="").validate_runtime_security()


def test_remote_public_origin_must_be_https_and_allowed_host() -> None:
    with pytest.raises(RuntimeError, match="https"):
        Settings(
            host="0.0.0.0", public_origin="http://agent.example", allowed_hosts=["agent.example"]
        ).validate_runtime_security()
    with pytest.raises(RuntimeError, match="ALLOWED_HOSTS"):
        Settings(host="0.0.0.0", public_origin="https://agent.example").validate_runtime_security()
    settings = Settings(
        host="0.0.0.0",
        public_origin="https://agent.example/",
        allowed_hosts=["agent.example"],
        extra_origins=[],
    )
    settings.validate_runtime_security()
    assert settings.allowed_origins() == {"https://agent.example"}
    assert settings.secure_cookies


def test_local_bind_works_without_token_or_origin() -> None:
    settings = Settings(host="127.0.0.1", port=8080, api_token="", public_origin="")
    settings.validate_runtime_security()
    assert "http://localhost:8080" in settings.allowed_origins()
    assert not settings.secure_cookies


def test_wildcard_hosts_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="exact host"):
        Settings(allowed_hosts=["*"]).validate_runtime_security()


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
