"""Shared skill defaults and the skill choices of each project.

Revision ID: 0023
Revises: 0022
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shared_skill_defaults",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "project_skill_settings",
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("scope", sa.String(10), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("project_id", "scope", "name", name="pk_project_skill_settings"),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
            name="fk_project_skill_settings_project_id",
        ),
        sa.CheckConstraint(
            "scope IN ('shared', 'project')", name="ck_project_skill_settings_scope"
        ),
    )


def downgrade() -> None:
    op.drop_table("project_skill_settings")
    op.drop_table("shared_skill_defaults")
