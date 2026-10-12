"""Encrypted model settings overrides.

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_provider_overrides",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("fields", sa.JSON(), nullable=False),
        sa.Column("api_key_ciphertext", sa.Text(), nullable=True),
        sa.Column("disabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "llm_model_overrides",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("fields", sa.JSON(), nullable=False),
        sa.Column("disabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "llm_routing_override",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("fields", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_llm_routing_singleton"),
    )


def downgrade() -> None:
    op.drop_table("llm_routing_override")
    op.drop_table("llm_model_overrides")
    op.drop_table("llm_provider_overrides")
