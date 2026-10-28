"""Add durable inter-session messages and terminal notice snapshots."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("run_generation", sa.Integer(), nullable=True))
    with op.batch_alter_table("sessions") as batch:
        batch.add_column(sa.Column("inbox_hop", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("notify_generation", sa.Integer(), nullable=True))
        batch.create_check_constraint("ck_sessions_inbox_hop", "inbox_hop >= 0")
        batch.create_check_constraint(
            "ck_sessions_notify_generation",
            "notify_generation IS NULL OR (notify_generation > 0 AND parent_id IS NOT NULL)",
        )
    with op.batch_alter_table("session_run_outcomes") as batch:
        batch.add_column(sa.Column("notice_target_id", sa.String(36), nullable=True))
        batch.add_column(sa.Column("notice_hop", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("notice_content", sa.Text(), nullable=True))
        batch.add_column(sa.Column("notice_message_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("notice_skipped_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_check_constraint(
            "ck_session_run_outcomes_notice",
            "(notice_target_id IS NULL AND notice_hop IS NULL AND notice_content IS NULL "
            "AND notice_message_id IS NULL AND notice_skipped_at IS NULL) OR "
            "(notice_target_id IS NOT NULL AND notice_hop IS NOT NULL "
            "AND notice_content IS NOT NULL AND notice_hop >= 0 "
            "AND length(notice_content) BETWEEN 1 AND 8000)",
        )
        batch.create_index(
            "ix_session_run_outcomes_notice",
            ["notice_skipped_at", "notice_target_id", "created_at", "session_id", "generation"],
        )
    op.create_table(
        "session_inbox",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "target_session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "sender_session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("hop", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("dedupe_key", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("kind IN ('message', 'child_finished')", name="ck_session_inbox_kind"),
        sa.CheckConstraint("length(content) BETWEEN 1 AND 8000", name="ck_session_inbox_content"),
        sa.CheckConstraint("hop >= 0", name="ck_session_inbox_hop"),
        sa.CheckConstraint(
            "NOT (delivered_at IS NOT NULL AND rejected_at IS NOT NULL)",
            name="ck_session_inbox_closed",
        ),
        sa.UniqueConstraint("dedupe_key", name="uq_session_inbox_dedupe_key"),
    )
    op.create_index(
        "ix_session_inbox_pending",
        "session_inbox",
        ["target_session_id", "delivered_at", "rejected_at", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_table("session_inbox")
    with op.batch_alter_table("session_run_outcomes") as batch:
        batch.drop_index("ix_session_run_outcomes_notice")
        batch.drop_constraint("ck_session_run_outcomes_notice", type_="check")
        batch.drop_column("notice_skipped_at")
        batch.drop_column("notice_message_id")
        batch.drop_column("notice_content")
        batch.drop_column("notice_hop")
        batch.drop_column("notice_target_id")
    with op.batch_alter_table("sessions") as batch:
        batch.drop_constraint("ck_sessions_notify_generation", type_="check")
        batch.drop_constraint("ck_sessions_inbox_hop", type_="check")
        batch.drop_column("notify_generation")
        batch.drop_column("inbox_hop")
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("run_generation")
