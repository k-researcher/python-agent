"""MCP servers that the owner adds in the UI.

Revision ID: 0024
Revises: 0023
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("command", sa.JSON(), nullable=False),
        sa.Column("env_secrets", sa.JSON(), nullable=False),
        sa.Column("project_ids", sa.JSON(), nullable=False),
        sa.Column("allowed_modes", sa.JSON(), nullable=False),
        sa.Column("risk_level", sa.String(30), nullable=False),
        sa.Column("allow_network", sa.Boolean(), nullable=False),
        sa.Column("read_paths", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_check", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("mcp_servers")
