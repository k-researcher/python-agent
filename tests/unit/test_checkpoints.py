"""Tests for loading context rows and persisting summary checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent.checkpoints import CheckpointError, load_context_rows, save_checkpoint
from agent.context_summary import SUMMARY_VERSION, SummaryCheckpoint
from agent.database import configure_sqlite_connection, engine, init_database, session_factory
from agent.models import Base, Message, Project, Session


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def make_session(tmp_path: Path, name: str = "checkpoints", subdir: str = "") -> str:
    """Create an isolated project and session, returning the session id."""
    await init_database()
    root_path = str(tmp_path / subdir) if subdir else str(tmp_path)
    async with session_factory() as db:
        project = Project(name=name, root_path=root_path)
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title=name, configuration={})
        db.add(session)
        await db.commit()
        return session.id


async def add_messages(session_id: str, *rows: dict[str, Any]) -> list[Message]:
    """Insert source messages and return them with their assigned ids."""
    async with session_factory() as db:
        messages = [Message(session_id=session_id, **row) for row in rows]
        db.add_all(messages)
        await db.commit()
        return messages


async def _save(
    session_id: str,
    text: str,
    covers_until: int,
    source_hash: str,
    *,
    model_id: str | None,
) -> int:
    checkpoint = SummaryCheckpoint(text=text, covers_until=covers_until, source_hash=source_hash)
    async with session_factory() as db:
        return await save_checkpoint(db, session_id, checkpoint, model_id)


async def add_summary(
    session_id: str,
    content: str,
    covers_until: int,
    source_hash: str,
    *,
    version: int = SUMMARY_VERSION,
) -> Message:
    """Insert a raw summary row directly, bypassing save-time validation."""
    async with session_factory() as db:
        message = Message(
            session_id=session_id,
            role="user",
            kind="summary",
            content=content,
            covers_until=covers_until,
            source_hash=source_hash,
            summary_version=version,
        )
        db.add(message)
        await db.commit()
        return message


async def test_without_checkpoint_returns_all_messages(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    messages = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "ok", "kind": "normal"},
    )
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [message.id for message in messages]


async def test_with_checkpoint_returns_anchors_and_tail(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "user", "content": "more", "kind": "normal"},
    )
    checkpoint_id = await _save(
        session_id, "## Goal\n- done", head[-1].id, _hash("prefix"), model_id="main"
    )
    tail = await add_messages(
        session_id,
        {"role": "user", "content": "tail-1", "kind": "normal"},
        {"role": "assistant", "content": "tail-2", "kind": "normal"},
    )
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    expected = [head[0].id, head[1].id, checkpoint_id, *[m.id for m in tail]]
    assert [row.id for row in rows] == expected
    assert [row.kind for row in rows] == ["normal", "normal", "summary", "normal", "normal"]


async def test_chooses_best_checkpoint_and_ignores_wrong_version(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "user", "content": "more", "kind": "normal"},
        {"role": "assistant", "content": "more-a", "kind": "normal"},
    )
    old = SummaryCheckpoint(
        text="old",
        covers_until=head[-1].id,
        source_hash=_hash("prev"),
        version=999,
    )
    async with session_factory() as db:
        await save_checkpoint(db, session_id, old, "main")
    small_id = await _save(session_id, "small", head[2].id, _hash("small"), model_id=None)
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    expected = [head[0].id, head[1].id, small_id, head[3].id, head[4].id]
    assert [row.id for row in rows] == expected


async def test_chooses_checkpoint_with_largest_coverage(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "user", "content": "more", "kind": "normal"},
    )
    big_id = await _save(session_id, "big", head[-1].id, _hash("big"), model_id="main")
    small_id = await _save(session_id, "small", head[1].id, _hash("small"), model_id="main")
    # The smaller-coverage checkpoint must win over the later id.
    assert small_id > big_id
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [head[0].id, head[1].id, big_id]


async def test_equal_coverage_keeps_latest_checkpoint(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
    )
    first_id = await _save(session_id, "one", head[-1].id, _hash("one"), model_id="main")
    second_id = await _save(session_id, "two", head[-1].id, _hash("two"), model_id="main")
    assert second_id != first_id
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [head[0].id, head[1].id, second_id]


async def test_duplicate_checkpoint_reuses_existing_row(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
    )
    checkpoint = SummaryCheckpoint(text="same", covers_until=head[-1].id, source_hash=_hash("x"))
    async with session_factory() as db:
        first_id = await save_checkpoint(db, session_id, checkpoint, "main")
    async with session_factory() as db:
        second_id = await save_checkpoint(db, session_id, checkpoint, "main")
    assert second_id == first_id
    async with session_factory() as db:
        count = await db.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.session_id == session_id, Message.kind == "summary")
        )
    assert count == 1


async def test_covers_until_from_another_session_raises(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path, name="source")
    other_id = await make_session(tmp_path, name="other", subdir="other")
    head = await add_messages(
        session_id,
        {"role": "user", "content": "task", "kind": "normal"},
    )
    other = await add_messages(
        other_id,
        {"role": "user", "content": "other", "kind": "normal"},
    )
    for bad in (head[-1].id + 1000, other[-1].id):
        checkpoint = SummaryCheckpoint(text="s", covers_until=bad, source_hash=_hash("h"))
        async with session_factory() as db:
            with pytest.raises(CheckpointError):
                await save_checkpoint(db, session_id, checkpoint, None)


@pytest.mark.parametrize(
    "bad_hash",
    ["short", "not-hex" * 8, "g" * 64],
)
async def test_invalid_source_hash_raises(tmp_path: Path, bad_hash: str) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "user", "content": "task", "kind": "normal"},
    )
    checkpoint = SummaryCheckpoint(text="s", covers_until=head[-1].id, source_hash=bad_hash)
    async with session_factory() as db:
        with pytest.raises(CheckpointError):
            await save_checkpoint(db, session_id, checkpoint, None)


async def test_empty_text_raises(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "user", "content": "task", "kind": "normal"},
    )
    checkpoint = SummaryCheckpoint(text="   ", covers_until=head[-1].id, source_hash=_hash("h"))
    async with session_factory() as db:
        with pytest.raises(CheckpointError):
            await save_checkpoint(db, session_id, checkpoint, None)


async def test_summary_model_none_is_saved(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "user", "content": "task", "kind": "normal"},
    )
    checkpoint_id = await _save(
        session_id, "## Goal\n- done", head[-1].id, _hash("free"), model_id=None
    )
    async with session_factory() as db:
        row = await db.get(Message, checkpoint_id)
    assert row is not None
    assert row.summary_model is None
    assert row.summary_version == SUMMARY_VERSION
    assert row.kind == "summary"


async def test_save_with_expire_on_commit_returns_id(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "user", "content": "task", "kind": "normal"},
    )
    checkpoint = SummaryCheckpoint(text="saved", covers_until=head[-1].id, source_hash=_hash("e"))
    async with AsyncSession(bind=engine, expire_on_commit=True) as db:
        checkpoint_id = await save_checkpoint(db, session_id, checkpoint, "main")
    assert isinstance(checkpoint_id, int)
    async with session_factory() as db:
        row = await db.get(Message, checkpoint_id)
        assert row is not None
        assert row.kind == "summary"
        assert row.content == "saved"


async def test_leading_system_stops_at_first_non_system(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys1", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "system", "content": "sys2", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "more-a", "kind": "normal"},
    )
    checkpoint_id = await _save(session_id, "sum", head[-1].id, _hash("s"), model_id=None)
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    # The system row after the assistant is not part of the leading prefix.
    assert [row.id for row in rows] == [head[0].id, head[3].id, checkpoint_id]


async def test_leading_system_without_normal_user(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys1", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "system", "content": "sys2", "kind": "normal"},
        {"role": "assistant", "content": "more-a", "kind": "normal"},
    )
    checkpoint_id = await _save(session_id, "sum", head[-1].id, _hash("s"), model_id=None)
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [head[0].id, checkpoint_id]


async def test_checkpoint_before_tail_keeps_projection_order(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "user", "content": "q2", "kind": "normal"},
        {"role": "assistant", "content": "a2", "kind": "normal"},
    )
    # The tail rows already exist; the checkpoint is saved afterwards and so gets
    # a higher id than every message it covers.
    checkpoint_id = await _save(session_id, "sum", head[1].id, _hash("p"), model_id=None)
    assert checkpoint_id > head[-1].id
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    expected = [head[0].id, head[1].id, checkpoint_id, head[2].id, head[3].id, head[4].id]
    assert [row.id for row in rows] == expected
    assert [row.kind for row in rows] == [
        "normal",
        "normal",
        "summary",
        "normal",
        "normal",
        "normal",
    ]


@pytest.mark.parametrize(
    ("bad_content", "bad_hash"),
    [
        ("   ", _hash("ok")),  # whitespace-only text, valid hash
        ("good text", "g" * 64),  # valid text, 64 non-hex chars
        ("   ", "g" * 64),  # both malformed
        ("good text", "a" * 65),  # valid text, too-long hash
    ],
)
async def test_unusable_checkpoint_does_not_displace_valid(
    tmp_path: Path, bad_content: str, bad_hash: str
) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "user", "content": "more", "kind": "normal"},
    )
    valid_id = await _save(session_id, "valid", head[1].id, _hash("valid"), model_id=None)
    # A later, larger-coverage checkpoint with malformed metadata must not win.
    await add_summary(session_id, bad_content, head[-1].id, bad_hash)
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [
        head[0].id,
        head[1].id,
        valid_id,
        head[2].id,
        head[3].id,
    ]


async def test_unusable_summary_coverage_skips_to_next_valid(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
        {"role": "user", "content": "more", "kind": "normal"},
    )
    valid_id = await _save(session_id, "valid", head[1].id, _hash("valid"), model_id=None)
    # The larger-coverage candidate covers a summary row, not a source message.
    await add_summary(session_id, "bad", valid_id, _hash("bad"))
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [
        head[0].id,
        head[1].id,
        valid_id,
        head[2].id,
        head[3].id,
    ]


async def test_unusable_other_session_coverage_is_skipped(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    other_session_id = await make_session(tmp_path, name="other", subdir="sub")
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
    )
    other = await add_messages(
        other_session_id,
        {"role": "user", "content": "other", "kind": "normal"},
    )
    valid_id = await _save(session_id, "valid", head[1].id, _hash("valid"), model_id=None)
    # The candidate has a later id, but its covered message belongs to another session.
    await add_summary(session_id, "stray", other[-1].id, _hash("stray"))
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [head[0].id, head[1].id, valid_id]


async def test_non_existent_coverage_falls_back_to_full_history(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "system", "content": "sys", "kind": "normal"},
        {"role": "user", "content": "task", "kind": "normal"},
        {"role": "assistant", "content": "a", "kind": "normal"},
    )
    # The only checkpoint covers an id that does not exist in this session, so it
    # is unusable and the raw history (including the stored summary row) returns.
    bad = await add_summary(session_id, "lost", 99999, _hash("lost"))
    async with session_factory() as db:
        rows = await load_context_rows(db, session_id)
    assert [row.id for row in rows] == [*[m.id for m in head], bad.id]


async def test_covers_until_referencing_summary_raises(tmp_path: Path) -> None:
    session_id = await make_session(tmp_path)
    head = await add_messages(
        session_id,
        {"role": "user", "content": "task", "kind": "normal"},
    )
    first_id = await _save(session_id, "first", head[-1].id, _hash("one"), model_id="main")
    checkpoint = SummaryCheckpoint(text="second", covers_until=first_id, source_hash=_hash("two"))
    async with session_factory() as db:
        with pytest.raises(CheckpointError):
            await save_checkpoint(db, session_id, checkpoint, "main")
    async with session_factory() as db:
        count = await db.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.session_id == session_id, Message.kind == "summary")
        )
    assert count == 1


async def _make_concurrent_engine(
    tmp_path: Path,
) -> tuple[async_sessionmaker[AsyncSession], AsyncEngine]:
    """Create an isolated SQLite engine with the production connection settings."""
    concurrent_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")

    @event.listens_for(concurrent_engine.sync_engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        configure_sqlite_connection(dbapi_connection)

    async with concurrent_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(concurrent_engine, expire_on_commit=False)
    return factory, concurrent_engine


async def test_concurrent_save_of_same_checkpoint(tmp_path: Path) -> None:
    factory, concurrent_engine = await _make_concurrent_engine(tmp_path)
    try:
        async with factory() as db:
            project = Project(name="concurrent", root_path=str(tmp_path / "project"))
            db.add(project)
            await db.flush()
            session = Session(project_id=project.id, title="concurrent", configuration={})
            db.add(session)
            await db.flush()
            message = Message(session_id=session.id, role="user", content="task", kind="normal")
            db.add(message)
            await db.flush()
            covered_id = message.id
            await db.commit()

        checkpoint = SummaryCheckpoint(text="same", covers_until=covered_id, source_hash=_hash("c"))

        async def save_once() -> int:
            async with factory() as db:
                return await save_checkpoint(db, session.id, checkpoint, "main")

        ids = await asyncio.gather(save_once(), save_once())
        assert ids[0] == ids[1]

        async with factory() as db:
            count = await db.scalar(
                select(func.count())
                .select_from(Message)
                .where(Message.session_id == session.id, Message.kind == "summary")
            )
        assert count == 1
    finally:
        await concurrent_engine.dispose()


class _CancellableSession(AsyncSession):
    """AsyncSession whose next commit is interrupted by a cancellation."""

    abort_next_commit: bool = False

    async def commit(self) -> None:
        if self.abort_next_commit:
            self.abort_next_commit = False
            raise asyncio.CancelledError()
        return await super().commit()


async def test_cancelled_save_leaves_no_row_and_db_usable(tmp_path: Path) -> None:
    factory, concurrent_engine = await _make_concurrent_engine(tmp_path)
    try:
        async with factory() as db:
            project = Project(name="cancel", root_path=str(tmp_path / "project"))
            db.add(project)
            await db.flush()
            session = Session(project_id=project.id, title="cancel", configuration={})
            db.add(session)
            await db.flush()
            message = Message(session_id=session.id, role="user", content="task", kind="normal")
            db.add(message)
            await db.flush()
            covered_id = message.id
            await db.commit()

        checkpoint = SummaryCheckpoint(
            text="never", covers_until=covered_id, source_hash=_hash("c")
        )
        cancellable_factory = async_sessionmaker(
            concurrent_engine, class_=_CancellableSession, expire_on_commit=False
        )
        # The flush (INSERT) happens, then the commit is cancelled; closing the
        # session must roll the transaction back so no half-written row remains.
        with pytest.raises(asyncio.CancelledError):
            async with cancellable_factory() as db:
                db.abort_next_commit = True
                await save_checkpoint(db, session.id, checkpoint, "main")

        async with factory() as db:
            count = await db.scalar(
                select(func.count())
                .select_from(Message)
                .where(Message.session_id == session.id, Message.kind == "summary")
            )
        assert count == 0

        # A later, normal save still works on the same database.
        async with factory() as db:
            saved_id = await save_checkpoint(db, session.id, checkpoint, "main")
            await db.commit()
        async with factory() as db:
            row = await db.get(Message, saved_id)
            assert row is not None
            assert row.content == "never"
    finally:
        await concurrent_engine.dispose()


async def test_concurrent_load_returns_stable_projection(tmp_path: Path) -> None:
    factory, concurrent_engine = await _make_concurrent_engine(tmp_path)
    try:
        async with factory() as db:
            project = Project(name="cload", root_path=str(tmp_path / "project"))
            db.add(project)
            await db.flush()
            session = Session(project_id=project.id, title="cload", configuration={})
            db.add(session)
            await db.flush()
            message = Message(session_id=session.id, role="user", content="task", kind="normal")
            db.add(message)
            await db.flush()
            covered_id = message.id
            checkpoint = Message(
                session_id=session.id,
                role="user",
                kind="summary",
                content="sum",
                covers_until=covered_id,
                source_hash=_hash("cl"),
                summary_version=SUMMARY_VERSION,
            )
            db.add(checkpoint)
            await db.commit()
            checkpoint_id = checkpoint.id

        async def load_once() -> list[int]:
            async with factory() as db:
                rows = await load_context_rows(db, session.id)
            return [row.id for row in rows]

        results = await asyncio.gather(load_once(), load_once())
        assert results[0] == results[1] == [message.id, checkpoint_id]
    finally:
        await concurrent_engine.dispose()
