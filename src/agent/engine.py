"""Neutral asynchronous port for an optional external session backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Health:
    status: Literal["ready", "degraded", "unavailable"]
    binary: str | None = None
    version: str | None = None
    supported: bool = False
    auth_kind: str | None = None
    isolation_capable: bool = False
    diagnostic: str | None = None


@dataclass(frozen=True)
class LimitWindow:
    name: str
    used_share: float | None
    reset_at: datetime | None
    duration_seconds: float | None = None
    status: str = "unknown"


@dataclass(frozen=True)
class Limits:
    windows: list[LimitWindow]
    status: str
    observed_at: datetime | None


@dataclass(frozen=True)
class EngineSpec:
    project_root: Path
    mode: Literal["read", "dev"]
    instructions: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class ThreadRef:
    native_id: str


@dataclass(frozen=True, kw_only=True)
class EventEnvelope:
    submission_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    item_id: str | None = None
    seq: int | None = None


@dataclass(frozen=True, kw_only=True)
class TextEvent(EventEnvelope):
    kind: Literal["text"] = field(default="text", init=False)
    text: str
    phase: Literal["delta", "final"]


@dataclass(frozen=True, kw_only=True)
class ReasoningEvent(EventEnvelope):
    kind: Literal["reasoning"] = field(default="reasoning", init=False)
    text: str
    phase: Literal["delta", "final"]


@dataclass(frozen=True, kw_only=True)
class ToolCallEvent(EventEnvelope):
    kind: Literal["tool_call"] = field(default="tool_call", init=False)
    name: str
    input: dict[str, Any]
    status: str


@dataclass(frozen=True, kw_only=True)
class ToolResultEvent(EventEnvelope):
    kind: Literal["tool_result"] = field(default="tool_result", init=False)
    related_item_id: str | None
    output: str
    success: bool


@dataclass(frozen=True, kw_only=True)
class DiffEvent(EventEnvelope):
    kind: Literal["diff"] = field(default="diff", init=False)
    paths: list[str]
    patch: str
    status: Literal["proposed", "applied"]


@dataclass(frozen=True, kw_only=True)
class UsageEvent(EventEnvelope):
    kind: Literal["usage"] = field(default="usage", init=False)
    input_tokens: int | None
    output_tokens: int | None
    cache_tokens: int | None
    reported_cost: Decimal | None
    estimated_cost: bool


@dataclass(frozen=True, kw_only=True)
class LimitsEvent(EventEnvelope):
    kind: Literal["limits"] = field(default="limits", init=False)
    limits: Limits


@dataclass(frozen=True, kw_only=True)
class ErrorEvent(EventEnvelope):
    kind: Literal["error"] = field(default="error", init=False)
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, kw_only=True)
class DoneEvent(EventEnvelope):
    kind: Literal["done"] = field(default="done", init=False)
    status: Literal["completed", "stopped", "error", "interrupted"]
    reason: str | None


@dataclass(frozen=True, kw_only=True)
class ApprovalEvent(EventEnvelope):
    kind: Literal["approval"] = field(default="approval", init=False)
    request_id: str
    action: str
    parameters: dict[str, Any]
    scope: Literal["request", "turn"]


type EngineEvent = (
    TextEvent
    | ReasoningEvent
    | ToolCallEvent
    | ToolResultEvent
    | DiffEvent
    | UsageEvent
    | LimitsEvent
    | ErrorEvent
    | DoneEvent
    | ApprovalEvent
)


@runtime_checkable
class Engine(Protocol):
    async def health(self) -> Health: ...
    async def open(self, spec: EngineSpec, thread_id: str | None = None) -> ThreadRef: ...
    def turn(
        self, thread: ThreadRef, input: str, submission_id: str
    ) -> AsyncIterator[EngineEvent]: ...
    async def answer(self, request_id: str, decision: Literal["allow", "deny"]) -> None: ...
    async def interrupt(self, turn_id: str) -> None: ...
    async def read_limits(self) -> Limits | None: ...
    async def close(self) -> None: ...
