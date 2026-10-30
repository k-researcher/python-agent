"""Persist native decision identity, scope, expiry and delivery."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("approvals") as batch:
        batch.add_column(sa.Column("generation", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("executor", sa.String(20), nullable=False, server_default="python")
        )
        batch.add_column(
            sa.Column("scope", sa.String(20), nullable=False, server_default="request")
        )
        batch.add_column(sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("approvals") as batch:
        for name in ("delivered_at", "expires_at", "scope", "executor", "generation"):
            batch.drop_column(name)
