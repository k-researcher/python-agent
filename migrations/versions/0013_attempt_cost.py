"""Снимок цены и точная стоимость каждой попытки запроса к LLM.

Ревизия: 0013
Предыдущая ревизия: 0012
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("outbound_audit") as batch:
        batch.add_column(sa.Column("pricing_snapshot", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("cost_amount", sa.String(100), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("outbound_audit") as batch:
        batch.drop_column("cost_amount")
        batch.drop_column("pricing_snapshot")
