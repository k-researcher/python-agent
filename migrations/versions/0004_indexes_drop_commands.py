"""Index session-scoped lookups and drop the unused commands table.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEXES: tuple[tuple[str, str, list[str]], ...] = (
    ("ix_sessions_parent_id", "sessions", ["parent_id"]),
    ("ix_sessions_archived_updated_at", "sessions", ["archived", "updated_at"]),
    ("ix_messages_session_id", "messages", ["session_id"]),
    ("ix_tool_calls_session_id", "tool_calls", ["session_id"]),
    ("ix_approvals_status", "approvals", ["status"]),
    ("ix_events_session_id", "events", ["session_id"]),
    ("ix_outbound_audit_session_id", "outbound_audit", ["session_id"]),
)


def upgrade() -> None:
    for name, table, columns in INDEXES:
        op.create_index(name, table, columns)
    op.drop_table("commands")


def downgrade() -> None:
    op.create_table(
        "commands",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("processed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    for name, table, _columns in reversed(INDEXES):
        op.drop_index(name, table_name=table)
