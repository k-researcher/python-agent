"""Context state: token calibration, summary checkpoints and LLM attempt audit.

Revision ID: 0008
Revises: 0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SUMMARY_CHECK = (
    "(kind = 'summary' AND role = 'user' AND content IS NOT NULL AND content <> ''"
    " AND covers_until > 0 AND summary_version > 0 AND source_hash IS NOT NULL)"
    " OR (kind <> 'summary' AND covers_until IS NULL AND source_hash IS NULL"
    " AND summary_version IS NULL AND summary_model IS NULL)"
)


def _audit_columns() -> list[sa.Column[object]]:
    """Return new column objects; a Column can belong to one table only."""
    return [
        sa.Column("inference_id", sa.String(36), nullable=True),
        sa.Column("purpose", sa.String(20), nullable=True),
        sa.Column("provider", sa.String(100), nullable=True),
        sa.Column("model_id", sa.String(100), nullable=True),
        sa.Column("model", sa.String(200), nullable=True),
        sa.Column("wire_version", sa.String(96), nullable=True),
        sa.Column("raw_estimate", sa.Integer(), nullable=True),
        sa.Column("factor", sa.Float(precision=53), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("error_kind", sa.String(30), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        "token_calibrations",
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("wire_version", sa.String(96), nullable=False),
        sa.Column("factor", sa.Float(precision=53), nullable=False, server_default="1.0"),
        sa.Column("samples", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("provider", "model", "wire_version", name="pk_token_calibrations"),
        sa.CheckConstraint("factor BETWEEN 0.5 AND 4.0", name="ck_token_calibrations_factor"),
        sa.CheckConstraint("samples >= 0", name="ck_token_calibrations_samples"),
    )
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("covers_until", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("source_hash", sa.String(64), nullable=True))
        batch.add_column(sa.Column("summary_version", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("summary_model", sa.String(100), nullable=True))
        batch.create_check_constraint("ck_messages_summary_fields", SUMMARY_CHECK)
        batch.create_unique_constraint(
            "uq_messages_summary_checkpoint",
            ["session_id", "covers_until", "source_hash", "summary_version"],
        )
    op.create_index("ix_messages_session_id_id", "messages", ["session_id", "id"])
    op.create_index(
        "ix_messages_session_kind_coverage",
        "messages",
        ["session_id", "kind", "covers_until", "id"],
    )
    op.create_index("ix_tool_calls_session_status", "tool_calls", ["session_id", "status"])
    with op.batch_alter_table("outbound_audit") as batch:
        for column in _audit_columns():
            batch.add_column(column)
    op.create_index("ix_outbound_audit_inference_id", "outbound_audit", ["inference_id"])


def downgrade() -> None:
    op.drop_index("ix_outbound_audit_inference_id", table_name="outbound_audit")
    with op.batch_alter_table("outbound_audit") as batch:
        for column in reversed(_audit_columns()):
            batch.drop_column(column.name)
    # Checkpoints are derived data: remove them, keep the source messages.
    op.execute("DELETE FROM messages WHERE kind = 'summary'")
    op.drop_index("ix_tool_calls_session_status", table_name="tool_calls")
    op.drop_index("ix_messages_session_kind_coverage", table_name="messages")
    op.drop_index("ix_messages_session_id_id", table_name="messages")
    with op.batch_alter_table("messages") as batch:
        batch.drop_constraint("uq_messages_summary_checkpoint", type_="unique")
        batch.drop_constraint("ck_messages_summary_fields", type_="check")
        batch.drop_column("summary_model")
        batch.drop_column("summary_version")
        batch.drop_column("source_hash")
        batch.drop_column("covers_until")
    op.drop_table("token_calibrations")
