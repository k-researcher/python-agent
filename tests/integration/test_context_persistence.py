from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import func, select

from agent.context_state import HistoryChanged, load_history, persist_checkpoint, source_item
from agent.context_summary import SUMMARY_VERSION, SummaryCheckpoint, source_hash
from agent.database import init_database, session_factory
from agent.models import Message, Project, Session


async def make_history(root: Path) -> tuple[str, list[int]]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name=root.name, root_path=str(root))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="context", status="running")
        db.add(session)
        await db.flush()
        rows = [
            Message(session_id=session.id, role="system", content="System anchor"),
            Message(session_id=session.id, role="user", content="First task"),
            Message(
                session_id=session.id,
                role="assistant",
                content="Old reply",
                reasoning_content="Raw reasoning",
            ),
            Message(
                session_id=session.id,
                role="assistant",
                content="Skipped",
                skipped=True,
                reasoning_effort="high",
            ),
            Message(session_id=session.id, role="assistant", content="Cut off", kind="interrupted"),
            Message(
                session_id=session.id,
                role="assistant",
                tool_calls=[
                    {
                        "id": "read",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                    }
                ],
            ),
            Message(
                session_id=session.id, role="tool", tool_call_id="read", content="Raw tool result"
            ),
            Message(session_id=session.id, role="user", content="Latest task"),
        ]
        db.add_all(rows)
        await db.commit()
        return session.id, [row.id for row in rows]


def candidate(boundary: int) -> SummaryCheckpoint:
    return SummaryCheckpoint("Earlier facts", boundary, "a" * 64, SUMMARY_VERSION)


async def test_checkpoint_is_durable_and_hashes_original_source(tmp_path: Path) -> None:
    session_id, ids = await make_history(tmp_path)
    async with session_factory() as db:
        history = await load_history(db, session_id)
        original = list(
            await db.scalars(
                select(Message).where(Message.session_id == session_id).order_by(Message.id)
            )
        )
        expected = source_hash([source_item(row) for row in original if row.id <= ids[-2]])
        original_fields = [source_item(row) for row in original]
    checkpoint_id = await persist_checkpoint(session_id, history, candidate(ids[-2]), "light")
    async with session_factory() as db:
        checkpoint = await db.get(Message, checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.source_hash == expected
        assert checkpoint.summary_model == "light"
        assert checkpoint.kind == "summary"
        reloaded = await load_history(db, session_id)
        assert reloaded.snapshot.checkpoint_message_id == checkpoint_id
        assert [row.id for row in reloaded.snapshot.messages] == [ids[0], ids[1], ids[-1]]
        assert reloaded.covered_messages == 3
        after = list(
            await db.scalars(
                select(Message)
                .where(Message.session_id == session_id, Message.kind != "summary")
                .order_by(Message.id)
            )
        )
        assert [source_item(row) for row in after] == original_fields


async def test_source_append_invalidates_prepared_checkpoint(tmp_path: Path) -> None:
    session_id, ids = await make_history(tmp_path)
    async with session_factory() as db:
        history = await load_history(db, session_id)
        db.add(Message(session_id=session_id, role="user", content="Concurrent input"))
        await db.commit()
    with pytest.raises(HistoryChanged):
        await persist_checkpoint(session_id, history, candidate(ids[-2]), None)
    async with session_factory() as db:
        assert (
            await db.scalar(
                select(func.count(Message.id)).where(
                    Message.session_id == session_id, Message.kind == "summary"
                )
            )
            == 0
        )


async def test_stop_prevents_checkpoint_write(tmp_path: Path) -> None:
    session_id, ids = await make_history(tmp_path)
    async with session_factory() as db:
        history = await load_history(db, session_id)
        session = await db.get(Session, session_id)
        assert session is not None
        session.stop_requested = True
        await db.commit()
    with pytest.raises(asyncio.CancelledError):
        await persist_checkpoint(session_id, history, candidate(ids[-2]), None)
    async with session_factory() as db:
        assert (
            await db.scalar(
                select(func.count(Message.id)).where(
                    Message.session_id == session_id, Message.kind == "summary"
                )
            )
            == 0
        )


@pytest.mark.parametrize("boundary_index", [0, 1, 5])
async def test_checkpoint_cannot_end_in_anchor_or_unfinished_exchange(
    tmp_path: Path, boundary_index: int
) -> None:
    session_id, ids = await make_history(tmp_path)
    async with session_factory() as db:
        history = await load_history(db, session_id)
    with pytest.raises(HistoryChanged):
        await persist_checkpoint(session_id, history, candidate(ids[boundary_index]), None)


async def test_stale_checkpoint_generation_requires_repreparation(tmp_path: Path) -> None:
    session_id, ids = await make_history(tmp_path)
    async with session_factory() as db:
        history = await load_history(db, session_id)
    await persist_checkpoint(session_id, history, candidate(ids[-2]), None)
    with pytest.raises(HistoryChanged):
        await persist_checkpoint(
            session_id, history, replace(candidate(ids[-2]), text="Other facts"), None
        )
    async with session_factory() as db:
        assert (
            await db.scalar(
                select(func.count(Message.id)).where(
                    Message.session_id == session_id, Message.kind == "summary"
                )
            )
            == 1
        )
