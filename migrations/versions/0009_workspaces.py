"""Пользователи и командные пространства.

Ревизия: 0009
Предыдущая ревизия: 0008
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BOOTSTRAP_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_WORKSPACE_ID = "00000000-0000-0000-0000-000000000002"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("is_operator", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "workspace_memberships",
        sa.Column(
            "workspace_id",
            sa.String(36),
            sa.ForeignKey(
                "workspaces.id",
                ondelete="CASCADE",
                name="fk_workspace_memberships_workspace_id",
            ),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey(
                "users.id",
                ondelete="CASCADE",
                name="fk_workspace_memberships_user_id",
            ),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "user_id", name="pk_workspace_memberships"),
        sa.CheckConstraint(
            "role IN ('owner', 'admin', 'member', 'viewer')",
            name="ck_workspace_memberships_role",
        ),
    )
    op.create_index(
        "uq_workspace_memberships_active_owner",
        "workspace_memberships",
        ["workspace_id"],
        unique=True,
        sqlite_where=sa.text("role = 'owner' AND revoked_at IS NULL"),
        postgresql_where=sa.text("role = 'owner' AND revoked_at IS NULL"),
    )

    now = datetime.now(UTC)
    users = sa.table(
        "users",
        sa.column("id", sa.String),
        sa.column("display_name", sa.String),
        sa.column("enabled", sa.Boolean),
        sa.column("is_operator", sa.Boolean),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    op.bulk_insert(
        users,
        [
            {
                "id": BOOTSTRAP_USER_ID,
                "display_name": "Owner",
                "enabled": True,
                "is_operator": True,
                "created_at": now,
            }
        ],
    )
    workspaces = sa.table(
        "workspaces",
        sa.column("id", sa.String),
        sa.column("name", sa.String),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    op.bulk_insert(
        workspaces,
        [
            {
                "id": DEFAULT_WORKSPACE_ID,
                "name": "Основное пространство",
                "created_at": now,
            }
        ],
    )
    memberships = sa.table(
        "workspace_memberships",
        sa.column("workspace_id", sa.String),
        sa.column("user_id", sa.String),
        sa.column("role", sa.String),
        sa.column("revoked_at", sa.DateTime(timezone=True)),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    op.bulk_insert(
        memberships,
        [
            {
                "workspace_id": DEFAULT_WORKSPACE_ID,
                "user_id": BOOTSTRAP_USER_ID,
                "role": "owner",
                "revoked_at": None,
                "created_at": now,
            }
        ],
    )

    with op.batch_alter_table("projects") as batch:
        batch.add_column(sa.Column("workspace_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_projects_workspace_id",
            "workspaces",
            ["workspace_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_index("ix_projects_workspace_id", "projects", ["workspace_id"])

    with op.batch_alter_table("sessions") as batch:
        batch.add_column(sa.Column("created_by_user_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_sessions_created_by_user_id",
            "users",
            ["created_by_user_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_index("ix_sessions_created_by_user_id", "sessions", ["created_by_user_id"])

    with op.batch_alter_table("auth_sessions") as batch:
        batch.add_column(sa.Column("user_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_auth_sessions_user_id",
            "users",
            ["user_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])

    with op.batch_alter_table("auth_bootstrap_codes") as batch:
        batch.add_column(sa.Column("user_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_auth_bootstrap_codes_user_id",
            "users",
            ["user_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_index("ix_auth_bootstrap_codes_user_id", "auth_bootstrap_codes", ["user_id"])

    op.execute(
        f"UPDATE projects SET workspace_id = '{DEFAULT_WORKSPACE_ID}' WHERE workspace_id IS NULL"
    )
    op.execute(
        f"UPDATE sessions SET created_by_user_id = '{BOOTSTRAP_USER_ID}' "
        "WHERE created_by_user_id IS NULL"
    )
    op.execute(f"UPDATE auth_sessions SET user_id = '{BOOTSTRAP_USER_ID}' WHERE user_id IS NULL")
    op.execute(
        f"UPDATE auth_bootstrap_codes SET user_id = '{BOOTSTRAP_USER_ID}' WHERE user_id IS NULL"
    )


def downgrade() -> None:
    op.drop_index("ix_auth_bootstrap_codes_user_id", table_name="auth_bootstrap_codes")
    with op.batch_alter_table("auth_bootstrap_codes") as batch:
        batch.drop_constraint("fk_auth_bootstrap_codes_user_id", type_="foreignkey")
        batch.drop_column("user_id")

    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    with op.batch_alter_table("auth_sessions") as batch:
        batch.drop_constraint("fk_auth_sessions_user_id", type_="foreignkey")
        batch.drop_column("user_id")

    op.drop_index("ix_sessions_created_by_user_id", table_name="sessions")
    with op.batch_alter_table("sessions") as batch:
        batch.drop_constraint("fk_sessions_created_by_user_id", type_="foreignkey")
        batch.drop_column("created_by_user_id")

    op.drop_index("ix_projects_workspace_id", table_name="projects")
    with op.batch_alter_table("projects") as batch:
        batch.drop_constraint("fk_projects_workspace_id", type_="foreignkey")
        batch.drop_column("workspace_id")

    op.drop_index("uq_workspace_memberships_active_owner", table_name="workspace_memberships")
    op.drop_table("workspace_memberships")
    op.drop_table("workspaces")
    op.drop_table("users")
