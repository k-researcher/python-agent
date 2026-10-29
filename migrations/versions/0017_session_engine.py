"""Optional personal engine backend, exclusive with the model backend."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("sessions") as batch:
        batch.add_column(sa.Column("engine_profile", sa.String(100), nullable=True))
        batch.add_column(sa.Column("engine_state", sa.JSON(), nullable=True))
        batch.alter_column("llm_profile", existing_type=sa.String(100), nullable=True)
        batch.create_check_constraint(
            "ck_sessions_backend",
            "(llm_profile IS NOT NULL AND length(trim(llm_profile)) > 0 AND engine_profile IS NULL)"
            " OR (llm_profile IS NULL AND engine_profile IS NOT NULL"
            " AND length(trim(engine_profile)) > 0)",
        )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM sessions WHERE engine_profile IS NOT NULL LIMIT 1"))
        .first()
    ):
        raise RuntimeError("Cannot downgrade while engine sessions exist")
    with op.batch_alter_table("sessions") as batch:
        batch.drop_constraint("ck_sessions_backend", type_="check")
        batch.alter_column("llm_profile", existing_type=sa.String(100), nullable=False)
        batch.drop_column("engine_state")
        batch.drop_column("engine_profile")
