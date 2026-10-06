from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class SessionStatus(enum.StrEnum):
    pending = "pending"
    running = "running"
    awaiting_confirmation = "awaiting_confirmation"
    completed = "completed"
    stopped = "stopped"
    error = "error"
    interrupted = "interrupted"


class ToolCallStatus(enum.StrEnum):
    pending = "pending"
    awaiting_confirmation = "awaiting_confirmation"
    running = "running"
    completed = "completed"
    rejected = "rejected"
    cancelled = "cancelled"
    error = "error"


class ApprovalStatus(enum.StrEnum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    cancelled = "cancelled"


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    root_path: Mapped[str] = mapped_column(String(4096), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sessions: Mapped[list[Session]] = relationship(back_populates="project")


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_archived_updated_at", "archived", "updated_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="RESTRICT"))
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), nullable=True, index=True
    )
    mode: Mapped[str] = mapped_column(String(50), default="dev")
    llm_profile: Mapped[str] = mapped_column(String(100), default="default", index=True)
    title: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(50), default=SessionStatus.pending.value)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    project: Mapped[Project] = relationship(back_populates="sessions")
    messages: Mapped[list[Message]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.id"
    )
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


# A summary checkpoint is a user message with coverage data; other kinds have none.
SUMMARY_FIELDS_CHECK = (
    "(kind = 'summary' AND role = 'user' AND content IS NOT NULL AND content <> ''"
    " AND covers_until > 0 AND summary_version > 0 AND source_hash IS NOT NULL)"
    " OR (kind <> 'summary' AND covers_until IS NULL AND source_hash IS NULL"
    " AND summary_version IS NULL AND summary_model IS NULL)"
)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(SUMMARY_FIELDS_CHECK, name="ck_messages_summary_fields"),
        UniqueConstraint(
            "session_id",
            "covers_until",
            "source_hash",
            "summary_version",
            name="uq_messages_summary_checkpoint",
        ),
        Index("ix_messages_session_id_id", "session_id", "id"),
        Index("ix_messages_session_kind_coverage", "session_id", "kind", "covers_until", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(30))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    # "normal", later "summary" and "interrupted". Only "normal" and "summary" reach the model.
    kind: Mapped[str] = mapped_column(String(20), default="normal")
    reasoning_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    # User message: the requested level ("auto" or an effort). Assistant: the level sent.
    reasoning_effort: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Summary checkpoint only: the last covered source message ID (inclusive), the hash of
    # the covered source, the checkpoint format version and the model profile (None: local).
    covers_until: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[Session] = relationship(back_populates="messages")


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_session_status", "session_id", "status"),)

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    risk_level: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default=ToolCallStatus.pending.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    session: Mapped[Session] = relationship(back_populates="tool_calls")
    approval: Mapped[Approval | None] = relationship(
        back_populates="tool_call", cascade="all, delete-orphan", uselist=False
    )


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tool_call_id: Mapped[str] = mapped_column(
        ForeignKey("tool_calls.id", ondelete="CASCADE"), unique=True
    )
    status: Mapped[str] = mapped_column(
        String(30), default=ApprovalStatus.pending.value, index=True
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tool_call: Mapped[ToolCall] = relationship(back_populates="approval")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    type: Mapped[str] = mapped_column(String(100))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OutboundAudit(Base):
    __tablename__ = "outbound_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    category: Mapped[str] = mapped_column(String(50))
    destination: Mapped[str] = mapped_column(String(2048))
    operation: Mapped[str] = mapped_column(String(100))
    payload_bytes: Mapped[int] = mapped_column(Integer, default=0)
    payload_sha256: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(30), default="started")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    # LLM attempt fields. One row covers one HTTP attempt: "started", then a terminal status.
    inference_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    purpose: Mapped[str | None] = mapped_column(String(20), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    wire_version: Mapped[str | None] = mapped_column(String(96), nullable=True)
    raw_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    factor: Mapped[float | None] = mapped_column(Float(precision=53), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_kind: Mapped[str | None] = mapped_column(String(30), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TokenCalibration(Base):
    """Correction factor of the token estimate for one provider, model and wire version."""

    __tablename__ = "token_calibrations"
    __table_args__ = (
        PrimaryKeyConstraint("provider", "model", "wire_version", name="pk_token_calibrations"),
        CheckConstraint("factor BETWEEN 0.5 AND 4.0", name="ck_token_calibrations_factor"),
        CheckConstraint("samples >= 0", name="ck_token_calibrations_samples"),
    )

    provider: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    wire_version: Mapped[str] = mapped_column(String(96))
    factor: Mapped[float] = mapped_column(Float(precision=53), default=1.0, server_default="1.0")
    samples: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    content: Mapped[str] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    keywords: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    document_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AuthSession(Base):
    """Browser login. Only the SHA-256 of the cookie value is stored."""

    __tablename__ = "auth_sessions"

    id_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    user_agent: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuthBootstrapCode(Base):
    """One-time login link code. Only the SHA-256 of the code is stored."""

    __tablename__ = "auth_bootstrap_codes"

    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
