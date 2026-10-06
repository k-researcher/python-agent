"""Reasoning text, requested reasoning level and kind for messages.

Revision ID: 0007
Revises: 0006
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.add_column(
            sa.Column("kind", sa.String(20), nullable=False, server_default="normal")
        )
        batch.add_column(sa.Column("reasoning_content", sa.Text(), nullable=True))
        batch.add_column(sa.Column("reasoning_effort", sa.String(20), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("reasoning_effort")
        batch.drop_column("reasoning_content")
        batch.drop_column("kind")
