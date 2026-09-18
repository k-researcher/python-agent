"""Add per-session LLM profile.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("llm_profile", sa.String(100), nullable=False, server_default="default"),
    )
    op.create_index("ix_sessions_llm_profile", "sessions", ["llm_profile"])


def downgrade() -> None:
    op.drop_index("ix_sessions_llm_profile", table_name="sessions")
    op.drop_column("sessions", "llm_profile")
