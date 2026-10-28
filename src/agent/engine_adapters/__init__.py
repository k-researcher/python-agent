"""Explicit profile factories; importing this package never imports an adapter."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import cast

from agent.config import EngineProfileSettings, Settings
from agent.engine import Engine
from agent.model_config import ModelConfigError, ModelsConfig, _Diagnostics, _parse_yaml
from agent.model_registry import models_source

EngineFactory = Callable[[EngineProfileSettings], Engine]


def configured_profiles(settings: Settings) -> dict[str, EngineProfileSettings]:
    """Read declarations without resolving API credentials or importing adapters."""
    path = models_source(settings)
    if path is None:
        profiles = dict(settings.engine_profiles)
    else:
        with path.open("rb") as stream:
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ModelConfigError("Models configuration exceeds the 1 MiB limit")
        profiles = ModelsConfig.model_validate(
            _parse_yaml(content, _Diagnostics(str(path)))
        ).engine_profiles
    if any(not 1 <= len(name) <= 100 or not name.strip() for name in profiles):
        raise ValueError("Engine profile names must contain 1-100 characters")
    if any(profile.process_limit > settings.max_parallel_sessions for profile in profiles.values()):
        raise ValueError("Engine process_limit exceeds max_parallel_sessions")
    return profiles


class EngineRegistry:
    def __init__(self, profiles: Mapping[str, EngineProfileSettings]) -> None:
        self.profiles = dict(profiles)
        self.factories: dict[str, EngineFactory] = {}

    def register(self, profile_id: str, factory: EngineFactory) -> None:
        """Register a factory for an explicitly configured profile (also used by fakes)."""
        if profile_id not in self.profiles:
            raise ValueError("Engine profile is not configured")
        self.factories[profile_id] = factory

    def get(self, profile_id: str) -> Engine | None:
        profile = self.profiles.get(profile_id)
        if profile is None or not profile.enabled:
            return None
        factory = self.factories.get(profile_id)
        if factory is None:
            try:
                module = importlib.import_module(profile.adapter_module)
            except ModuleNotFoundError:
                return None
            factory = cast(EngineFactory, module.Engine)
            self.factories[profile_id] = factory
        return factory(profile)
