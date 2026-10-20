"""Ревизия источника вызова инструмента.

Ревизия: 0012
Предыдущая ревизия: 0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tool_calls") as batch:
        batch.add_column(sa.Column("source_revision", sa.String(64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tool_calls") as batch:
        batch.drop_column("source_revision")
