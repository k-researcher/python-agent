"""Durable submission journal and observed external tool identity."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "engine_turns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column(
            "input_message_id",
            sa.Integer(),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("submission_id", sa.String(36), nullable=False, unique=True),
        sa.Column("native_turn_id", sa.String(200), nullable=True),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("last_item_ids", sa.JSON(), nullable=False),
        sa.Column("outcome", sa.JSON(), nullable=True),
        sa.UniqueConstraint(
            "session_id", "generation", "input_message_id", name="uq_engine_turns_input"
        ),
        sa.CheckConstraint(
            "state IN ('prepared', 'dispatched', 'done')", name="ck_engine_turns_state"
        ),
    )
    with op.batch_alter_table("tool_calls") as batch:
        batch.add_column(
            sa.Column("executor", sa.String(20), nullable=False, server_default="python")
        )
        batch.add_column(sa.Column("native_item_id", sa.String(200), nullable=True))
        batch.add_column(sa.Column("native_request_id", sa.String(200), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tool_calls") as batch:
        batch.drop_column("native_request_id")
        batch.drop_column("native_item_id")
        batch.drop_column("executor")
    op.drop_table("engine_turns")
