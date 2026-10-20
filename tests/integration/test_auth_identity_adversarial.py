from __future__ import annotations

import asyncio
import hashlib

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.auth import IssuedSession, authenticate, consume_login_code, issue_login_code
from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import AuthBootstrapCode, AuthSession, User, utcnow


async def _bind_code(db: AsyncSession, code: str, user_id: str) -> None:
    digest = hashlib.sha256(code.encode()).hexdigest()
    await db.execute(
        update(AuthBootstrapCode)
        .where(AuthBootstrapCode.code_hash == digest)
        .values(user_id=user_id)
    )


async def _session_user_id(raw_id: str) -> str | None:
    async with session_factory() as db:
        auth = await authenticate(db, raw_id)
        return auth.user_id if auth is not None else None


async def test_reverse_order_consumption_keeps_identity_isolated() -> None:
    await init_database(recover_running=False)
    settings = Settings()
    async with session_factory() as db:
        first = User(display_name="first")
        second = User(display_name="second")
        db.add_all([first, second])
        await db.commit()

        code_first = await issue_login_code(db, settings)
        code_second = await issue_login_code(db, settings)
        await _bind_code(db, code_first, first.id)
        await _bind_code(db, code_second, second.id)
        await db.commit()

    async with session_factory() as db:
        issued_second = await consume_login_code(db, settings, code_second, "reverse-first")
        issued_first = await consume_login_code(db, settings, code_first, "reverse-second")
    assert issued_second is not None
    assert issued_first is not None
    assert issued_second.raw_id != issued_first.raw_id

    first_user_id = await _session_user_id(issued_first.raw_id)
    second_user_id = await _session_user_id(issued_second.raw_id)
    assert first_user_id is not None and second_user_id is not None
    assert first_user_id != second_user_id
    assert first_user_id == first.id
    assert second_user_id == second.id


async def test_consumed_at_revokes_login_link_for_bound_code() -> None:
    await init_database(recover_running=False)
    settings = Settings()
    async with session_factory() as db:
        user = User(display_name="revoked")
        db.add(user)
        await db.commit()
        code = await issue_login_code(db, settings)
        await _bind_code(db, code, user.id)
        await db.commit()

        digest = hashlib.sha256(code.encode()).hexdigest()
        before = await db.scalar(select(func.count()).select_from(AuthSession))

        await db.execute(
            update(AuthBootstrapCode)
            .where(AuthBootstrapCode.code_hash == digest)
            .values(consumed_at=utcnow())
        )
        await db.commit()
        assert await consume_login_code(db, settings, code, "revoked-replay") is None
        assert await db.scalar(select(func.count()).select_from(AuthSession)) == before


async def test_concurrent_consumption_of_same_bound_code_creates_one_session() -> None:
    await init_database(recover_running=False)
    settings = Settings()
    async with session_factory() as db:
        user = User(display_name="racing")
        db.add(user)
        await db.commit()
        code = await issue_login_code(db, settings)
        await _bind_code(db, code, user.id)
        await db.commit()

    async with session_factory() as db:
        before = await db.scalar(select(func.count()).select_from(AuthSession))

    async def consume(tag: str) -> IssuedSession | None:
        async with session_factory() as db:
            return await consume_login_code(db, settings, code, tag)

    first, second = await asyncio.gather(consume("racers-1"), consume("racers-2"))
    winners = [issued for issued in (first, second) if issued is not None]
    assert len(winners) == 1

    async with session_factory() as db:
        assert before is not None
        assert await db.scalar(select(func.count()).select_from(AuthSession)) == before + 1
        auth = await authenticate(db, winners[0].raw_id)
        assert auth is not None and auth.user_id == user.id
