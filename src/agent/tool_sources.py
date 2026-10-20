"""Read immutable tool-source definitions from trusted server configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from agent.tools.base import RiskLevel

_MAX_FILE_BYTES = 1 << 20
_MAX_SOURCES = 100
_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ToolSourceError(ValueError):
    """Report invalid configuration without including its content or location."""


def _check_id(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("Invalid ID")
    return value


class ToolSource(BaseModel):
    """Describe an operator-controlled source, disabled until explicitly enabled.

    Environment entries are variable names, not secret values. This model does
    not execute commands, authenticate callers, or grant tool approvals.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    workspace_id: str
    project_ids: frozenset[str] = frozenset()
    command: tuple[str, ...]
    env_keys: tuple[str, ...] = ()
    enabled: StrictBool = False
    risk_level: RiskLevel = RiskLevel.network_access
    allowed_modes: frozenset[Literal["dev", "ask"]] = frozenset({"dev"})
    max_result_chars: StrictInt = Field(default=20_000, gt=0, le=1_000_000)

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        if not _SOURCE_ID.fullmatch(value):
            raise ValueError("Invalid source ID")
        return value

    @field_validator("workspace_id")
    @classmethod
    def _workspace_id(cls, value: str) -> str:
        return _check_id(value)

    @field_validator("project_ids")
    @classmethod
    def _project_ids(cls, value: frozenset[str]) -> frozenset[str]:
        for project_id in value:
            _check_id(project_id)
        return value

    @field_validator("command")
    @classmethod
    def _command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("Command must not be empty")
        for argument in value:
            _check_id(argument)
        return value

    @field_validator("env_keys")
    @classmethod
    def _env_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _ENV_KEY.fullmatch(key) for key in value):
            raise ValueError("Invalid environment variable name")
        return value


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ToolSourceError("Duplicate configuration field")
        result[key] = value
    return result


def load_tool_sources(path: Path | None) -> tuple[ToolSource, ...]:
    """Load a bounded JSON document without launching sources or reading secrets.

    Only the trusted server selects the path. No project discovery is performed.
    Invalid files raise ToolSourceError with a redacted message and traceback.
    """
    if path is None:
        return ()
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
            raise ToolSourceError("Tool-source configuration is too large")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
        if not isinstance(data, dict) or set(data) != {"sources"}:
            raise ToolSourceError("Invalid tool-source configuration")
        entries = data["sources"]
        if not isinstance(entries, list) or len(entries) > _MAX_SOURCES:
            raise ToolSourceError("Invalid tool-source list")
        sources = tuple(ToolSource.model_validate(entry) for entry in entries)
        identities = {(source.workspace_id, source.source_id) for source in sources}
        if len(identities) != len(sources):
            raise ToolSourceError("Duplicate tool source")
        return sources
    except (OSError, ValueError, RecursionError):
        raise ToolSourceError("Invalid tool-source configuration") from None
