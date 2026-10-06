from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.sandbox import SandboxPolicy


class RiskLevel(enum.StrEnum):
    read_only = "read_only"
    local_write = "local_write"
    process_execution = "process_execution"
    network_access = "network_access"
    remote_write = "remote_write"
    destructive = "destructive"
    secret_access = "secret_access"


AuditCallback = Callable[[str, str, str, str, str, str | None], Awaitable[None]]
SpawnChildCallback = Callable[[str, str, bool, str | None], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    project_root: Path
    depth: int = 0
    audit_egress: AuditCallback | None = None
    spawn_child: SpawnChildCallback | None = None
    sandbox: SandboxPolicy | None = None


class Tool(ABC):
    name: str
    description: str
    risk_level: RiskLevel
    input_schema: dict[str, Any]
    allowed_modes: frozenset[str] = frozenset({"dev", "ask"})
    network_capability: bool = False
    knowledge_capability: bool = False

    def parameters(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """JSON Schema sent to the model; override when it depends on live configuration."""
        return self.input_schema

    @abstractmethod
    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ToolError(RuntimeError):
    pass
