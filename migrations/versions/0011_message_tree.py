"""Родительские связи дерева сообщений.

Ревизия: 0011
Предыдущая ревизия: 0010
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("parent_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_messages_parent_id_messages",
            "messages",
            ["parent_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_messages_session_parent_id", "messages", ["session_id", "parent_id"])

    messages = sa.table(
        "messages",
        sa.column("id", sa.Integer()),
        sa.column("session_id", sa.String()),
        sa.column("parent_id", sa.Integer()),
        sa.column("kind", sa.String()),
        sa.column("skipped", sa.Boolean()),
    )
    previous = messages.alias("previous")
    parent = (
        sa.select(sa.func.max(previous.c.id))
        .where(
            previous.c.session_id == messages.c.session_id,
            previous.c.id < messages.c.id,
            previous.c.kind == "normal",
            previous.c.skipped.is_(False),
        )
        .correlate(messages)
        .scalar_subquery()
    )
    op.execute(
        sa.update(messages)
        .where(messages.c.kind == "normal", messages.c.skipped.is_(False))
        .values(parent_id=parent)
    )


def downgrade() -> None:
    op.drop_index("ix_messages_session_parent_id", table_name="messages")
    with op.batch_alter_table("messages") as batch:
        batch.drop_constraint("fk_messages_parent_id_messages", type_="foreignkey")
        batch.drop_column("parent_id")
