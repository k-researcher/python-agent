"""Build the effective model registry for the running application."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import PROJECT_ROOT, Settings, _env_file
from agent.model_config import (
    ModelConfigError,
    ResolvedModel,
    ResolvedModelRegistry,
    load_models,
)

DEFAULT_MODELS_FILE = PROJECT_ROOT / "config" / "models.yaml"
logger = logging.getLogger(__name__)


def process_env() -> dict[str, str]:
    """Selected dotenv overlaid by the process environment (process wins)."""
    env_file = _env_file()
    env = (
        {name: value for name, value in dotenv_values(env_file).items() if value is not None}
        if env_file is not None and env_file.is_file()
        else {}
    )
    env.update(os.environ)
    return env


def models_source(settings: Settings) -> Path | None:
    """Return the YAML file to read, or None for the legacy AGENT_LLM_* settings.

    - Not set: config/models.yaml when it exists, otherwise legacy.
    - Empty string: legacy, explicitly.
    - A path: the file must exist. A typo must not send data to an unexpected provider.
    """
    value = settings.models_file
    if value is None:
        return DEFAULT_MODELS_FILE if DEFAULT_MODELS_FILE.is_file() else None
    if not str(value).strip():
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise ModelConfigError(f"{path}:1:1: <root>: configuration file is missing")
    return path


def load_registry(settings: Settings) -> ResolvedModelRegistry:
    """Read YAML or, when there is no YAML source, the legacy AGENT_LLM_* settings."""
    path = models_source(settings)
    return load_models(path, process_env(), settings if path is None else None)


def select_model(
    registry: ResolvedModelRegistry,
    requested: str | None,
    role: str | None = None,
    *,
    strict: bool = True,
) -> ResolvedModel:
    """Explicit model, then the role's default, then the global default.

    With ``strict=False`` an unknown model ID gives the default model. Existing sessions use
    this: a change from legacy settings to YAML can remove their model.
    """
    model_id = requested or (registry.roles.get(role) if role else None) or registry.default_id
    if model_id not in registry.models and not strict:
        logger.warning("Model %r is not configured; the default model is used", model_id)
        model_id = registry.default_id
    try:
        return registry.get(model_id)
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_id}") from exc


async def effective_registry(db: AsyncSession, settings: Settings) -> ResolvedModelRegistry:
    """YAML (or legacy) merged with the overrides saved from the UI."""
    from agent.model_overrides import load_effective_registry

    return await load_effective_registry(db, settings)
