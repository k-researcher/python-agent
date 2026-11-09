"""MCP servers: how long one tool call may wait for the server.

Revision ID: 0026
Revises: 0025
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "mcp_servers",
        sa.Column("call_timeout", sa.Integer(), nullable=False, server_default="120"),
    )


def downgrade() -> None:
    op.drop_column("mcp_servers", "call_timeout")
