"""Per-model inference safety limits, including legacy resolution."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.config import Settings
from agent.model_config import ModelSpec, load_models
from tests.inference_guard_helpers import guard_tasks as guard_tasks

pytestmark = pytest.mark.usefixtures("guard_tasks")


def spec(**changes):
    return ModelSpec(
        provider="gateway", model="engine", context_window=8192, max_tokens=1024, **changes
    )


async def test_defaults_override_and_off(tmp_path: Path):
    async with asyncio.timeout(5):
        model = spec()
        assert model.inference_timeout_seconds == 600
        assert model.semantic_idle_timeout_seconds == 120
        assert model.repetition_threshold_chars == 65536
        assert (
            spec(
                inference_timeout_seconds=300,
                semantic_idle_timeout_seconds=20,
                repetition_threshold_chars=None,
            ).repetition_threshold_chars
            is None
        )
        legacy = load_models(tmp_path / "missing", {}, Settings()).get("default")
        assert legacy.inference_timeout_seconds == 600
        assert legacy.semantic_idle_timeout_seconds == 120
        assert legacy.repetition_threshold_chars == 65536


@pytest.mark.parametrize("field", ["inference_timeout_seconds", "semantic_idle_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "120", True, None])
async def test_invalid_time_limit(field, value):
    async with asyncio.timeout(5):
        with pytest.raises(ValidationError):
            spec(**{field: value})


@pytest.mark.parametrize("value", [0, -1, 1.5, "32", True, float("inf")])
async def test_invalid_repetition_limit(value):
    async with asyncio.timeout(5):
        with pytest.raises(ValidationError):
            spec(repetition_threshold_chars=value)


async def test_idle_cannot_exceed_total():
    async with asyncio.timeout(5):
        with pytest.raises(ValidationError):
            spec(inference_timeout_seconds=30, semantic_idle_timeout_seconds=31)


async def test_yaml_resolution_preserves_overrides_and_off(tmp_path):
    async with asyncio.timeout(5):
        path = tmp_path / "models.yaml"
        path.write_text("""providers:
  gateway:
    kind: openai
    base_url: https://example.test/v1
models:
  main:
    provider: gateway
    model: engine
    context_window: 8192
    max_tokens: 1024
    inference_timeout_seconds: 300
    semantic_idle_timeout_seconds: 30
    repetition_threshold_chars: null
routing:
  default: main
""")
        model = load_models(path, {}, None).get("main")
        assert model.inference_timeout_seconds == 300
        assert model.semantic_idle_timeout_seconds == 30
        assert model.repetition_threshold_chars is None
