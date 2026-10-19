from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from agent.auth import authenticate, consume_login_code, issue_login_code
from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import AuthBootstrapCode, AuthSession, User, utcnow


@pytest.mark.parametrize("bound", [True, False])
async def test_login_preserves_only_explicit_identity(bound: bool) -> None:
    await init_database(recover_running=False)
    settings = Settings()
    async with session_factory() as db:
        user = User(display_name="Explicit login identity")
        db.add(user)
        await db.commit()
        code = await issue_login_code(db, settings)
        if bound:
            await db.execute(
                update(AuthBootstrapCode)
                .where(AuthBootstrapCode.code_hash == hashlib.sha256(code.encode()).hexdigest())
                .values(user_id=user.id)
            )
            await db.commit()
        issued = await consume_login_code(db, settings, code, "identity-test")
        assert issued is not None
        auth = await authenticate(db, issued.raw_id)
        assert auth is not None
        assert auth.user_id == (user.id if bound else None)
        before = await db.scalar(select(func.count()).select_from(AuthSession))
        assert await consume_login_code(db, settings, code, "replay") is None
        assert await db.scalar(select(func.count()).select_from(AuthSession)) == before


async def test_login_reads_identity_from_database_not_orm_cache() -> None:
    await init_database(recover_running=False)
    settings = Settings()
    async with session_factory() as db:
        previous = User(display_name="Previous identity")
        current = User(display_name="Current identity")
        db.add_all([previous, current])
        await db.commit()
        code = await issue_login_code(db, settings)
        digest = hashlib.sha256(code.encode()).hexdigest()
        await db.execute(
            update(AuthBootstrapCode)
            .where(AuthBootstrapCode.code_hash == digest)
            .values(user_id=previous.id)
        )
        await db.commit()
        cached = await db.get(AuthBootstrapCode, digest)
        assert cached is not None and cached.user_id == previous.id
        async with session_factory() as other:
            await other.execute(
                update(AuthBootstrapCode)
                .where(AuthBootstrapCode.code_hash == digest)
                .values(user_id=current.id)
            )
            await other.commit()
        issued = await consume_login_code(db, settings, code, "identity-test")
        assert issued is not None
        auth = await authenticate(db, issued.raw_id)
        assert auth is not None and auth.user_id == current.id


async def test_expired_bound_code_does_not_create_session() -> None:
    await init_database(recover_running=False)
    settings = Settings()
    async with session_factory() as db:
        user = User(display_name="Expired identity")
        db.add(user)
        await db.commit()
        code = await issue_login_code(db, settings)
        await db.execute(
            update(AuthBootstrapCode)
            .where(AuthBootstrapCode.code_hash == hashlib.sha256(code.encode()).hexdigest())
            .values(user_id=user.id, expires_at=utcnow() - timedelta(seconds=1))
        )
        await db.commit()
        before = await db.scalar(select(func.count()).select_from(AuthSession))
        assert await consume_login_code(db, settings, code, "expired") is None
        assert await db.scalar(select(func.count()).select_from(AuthSession)) == before
