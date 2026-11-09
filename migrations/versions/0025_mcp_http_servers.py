"""MCP servers over HTTP: the address, the secret headers and the protocol era.

Revision ID: 0025
Revises: 0024
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("mcp_servers", sa.Column("url", sa.String(2048), nullable=True))
    op.add_column(
        "mcp_servers",
        sa.Column("header_secrets", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "mcp_servers",
        sa.Column("http_protocol", sa.String(10), nullable=False, server_default="auto"),
    )


def downgrade() -> None:
    op.drop_column("mcp_servers", "http_protocol")
    op.drop_column("mcp_servers", "header_secrets")
    op.drop_column("mcp_servers", "url")
