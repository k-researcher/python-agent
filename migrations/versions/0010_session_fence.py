"""Поля защиты сессии от устаревшего исполнителя.

Ревизия: 0010
Предыдущая ревизия: 0009
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.add_column(
            sa.Column("run_generation", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("owner_epoch", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("owner_token", sa.String(36), nullable=True))
        batch.add_column(sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.drop_column("lease_until")
        batch.drop_column("owner_token")
        batch.drop_column("owner_epoch")
        batch.drop_column("run_generation")
