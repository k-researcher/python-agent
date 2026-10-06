from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    root_path: str = Field(min_length=1, max_length=4096)


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    root_path: str
    created_at: datetime


class SessionCreate(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1, max_length=2_000_000)
    mode: Literal["dev", "ask"] = "dev"
    llm_profile: str | None = Field(default=None, min_length=1, max_length=100)
    reasoning_effort: str | None = Field(default=None, max_length=20)
    title: str | None = Field(default=None, max_length=300)
    auto_start: bool = True


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
    created_at: datetime
    updated_at: datetime


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
    created_at: datetime


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
    created_at: datetime


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str | None
    type: str
    payload: dict[str, Any]
    created_at: datetime


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
    created_at: datetime


class KnowledgeDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    content: str
    content_sha256: str
    keywords: str
    document_metadata: dict[str, Any]
    source_session_id: str | None
    created_at: datetime
    updated_at: datetime
