from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def _assume_utc(value: datetime) -> datetime:
    """SQLite drops the time zone: a stored value without one is UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


# Response time stamps always carry the time zone, so a browser does not read them as local time.
UtcDatetime = Annotated[datetime, AfterValidator(_assume_utc)]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    root_path: str = Field(min_length=1, max_length=4096)


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    root_path: str
    created_at: UtcDatetime


class SessionBudgetRequest(BaseModel):
    """Optional per-session limits; each value must be positive and finite."""

    max_steps: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost: str | None = None
    deadline_seconds: float | None = Field(default=None, gt=0)

    @field_validator("max_cost")
    @classmethod
    def _validate_max_cost(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("max_cost must be a decimal string") from exc
        if not parsed.is_finite() or parsed <= 0:
            raise ValueError("max_cost must be a positive finite decimal")
        return value


class SessionBudgetRead(BaseModel):
    """Durable limits and consumed usage of a session budget."""

    max_steps: int
    max_tokens: int | None
    max_cost: str | None
    deadline_seconds: float | None
    steps: int
    known_tokens: int
    known_cost: str
    has_unknown_tokens: bool
    has_unknown_cost: bool
    started_at: UtcDatetime | None
    stop_reason: str | None


class SessionCreate(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1, max_length=2_000_000)
    mode: Literal["dev", "ask"] = "dev"
    llm_profile: str | None = Field(default=None, min_length=1, max_length=100)
    reasoning_effort: str | None = Field(default=None, max_length=20)
    title: str | None = Field(default=None, max_length=300)
    auto_start: bool = True
    budget: SessionBudgetRequest | None = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    parent_id: str | None
    mode: str
    llm_profile: str
    configuration: dict[str, Any]
    title: str
    status: str
    error: str | None
    archived: bool
    # Populated by agent.api; the message-tree fork route returns it as None.
    budget: SessionBudgetRead | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=2_000_000)
    # Level for this message and the tool loop that follows it: "auto" or a model level.
    reasoning_effort: str | None = Field(default=None, max_length=20)


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    role: str
    content: str | None
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]] | None
    token_count: int | None
    skipped: bool
    kind: str
    reasoning_content: str | None
    reasoning_effort: str | None
    covers_until: int | None
    source_hash: str | None
    summary_version: int | None
    summary_model: str | None
    created_at: UtcDatetime


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    comment: str | None = Field(default=None, max_length=10_000)


class ApprovalRead(BaseModel):
    id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str
    status: str
    comment: str | None
    created_at: UtcDatetime


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str | None
    type: str
    payload: dict[str, Any]
    created_at: UtcDatetime


class OutboundAuditRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str | None
    category: str
    destination: str
    operation: str
    payload_bytes: int
    payload_sha256: str
    status: str
    detail: str | None
    created_at: UtcDatetime


class KnowledgeDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    content: str
    content_sha256: str
    keywords: str
    document_metadata: dict[str, Any]
    source_session_id: str | None
    created_at: UtcDatetime
    updated_at: UtcDatetime
