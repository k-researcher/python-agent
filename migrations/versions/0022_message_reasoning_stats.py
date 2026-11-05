"""Reasoning token count and duration of an assistant message.

Revision ID: 0022
Revises: 0021
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("reasoning_stats", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "reasoning_stats")
