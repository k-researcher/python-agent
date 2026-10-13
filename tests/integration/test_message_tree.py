"""Integration tests for the message tree core and its invariants.

Covers ``src/agent/message_tree.py`` (append, edit, regenerate, switch_branch,
fork) and its interaction with the session fence (``session_fence.py``) and the
context projection (``context_projection.py``/``checkpoints.py``).

The central invariant under test: the active ``normal`` rows of a session always
form one root→leaf chain linked by ``parent_id``, with strictly increasing ids,
so context building, summaries and checkpoints remain well-defined.

The migrated schema already carries the ``parent_id`` column (revision 0011).
Instantiating the session_factory reuses a single SQLite database inside the
test temp dir, isolated by ``tests/conftest.py``.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, update

from agent.checkpoints import save_checkpoint
from agent.config import get_settings
from agent.context_projection import (
    HistorySnapshot,
    SourceMessage,
    build_snapshot,
    chain_hash,
    project,
)
from agent.context_summary import SUMMARY_VERSION, SummaryCheckpoint
from agent.database import init_database, session_factory
from agent.message_tree import (
    TreeError,
    append_message,
    edit_message,
    fork,
    path_to,
    regenerate,
    switch_branch,
)
from agent.models import (
    Approval,
    ApprovalStatus,
    Message,
    Project,
    Session,
    SessionStatus,
    ToolCall,
    ToolCallStatus,
)
from agent.session_fence import (
    FenceLostError,
    RunFence,
    claim,
    fence_check,
    stop_generation,
)

_HEX = frozenset("0123456789abcdefABCDEF")


async def new_session(root: Path, **values: Any) -> str:
    """Create a fresh project and an idle (completed) session; return its id."""
    await init_database(recover_running=False)
    tag = uuid.uuid4().hex[:8]
    async with session_factory() as db:
        project = Project(name=f"tree-{tag}", root_path=str(root / tag))
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id,
            title="tree",
            status=SessionStatus.completed.value,
            configuration={},
            **values,
        )
        db.add(session)
        await db.commit()
        return session.id


async def read_rows(session_id: str) -> list[Message]:
    """Return all session rows ordered by id."""
    async with session_factory() as db:
        return list(
            await db.scalars(
                select(Message).where(Message.session_id == session_id).order_by(Message.id)
            )
        )


async def active_rows(session_id: str) -> list[Message]:
    """Return the active normal rows ordered by id."""
    async with session_factory() as db:
        rows = list(
            await db.scalars(
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.kind == "normal",
                    Message.skipped.is_(False),
                )
                .order_by(Message.id)
            )
        )
    return rows


async def assert_chain_invariant(session_id: str) -> list[Message]:
    """Assert active normal rows form one root→leaf chain; return them by id."""
    rows = await active_rows(session_id)
    if not rows:
        return rows
    assert rows[0].parent_id is None, "the root message must have no parent"
    for previous, current in zip(rows, rows[1:], strict=False):
        assert current.parent_id == previous.id, "active rows must be linked by parent_id"
        assert current.id > previous.id, "active ids must increase along the chain"
        assert current.kind == "normal"
    return rows


# --------------------------------------------------------------------------- #
# 1. append_message: chaining inside one transaction, summary never head
# --------------------------------------------------------------------------- #


async def test_append_message_chains_in_one_transaction_and_summary_never_head(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path)
    ids: list[int] = []
    summary_id = 0
    async with session_factory() as db:
        first = await append_message(db, session_id, role="user", content="first")
        assert first.parent_id is None
        ids.append(first.id)
        for index in range(3):
            row = await append_message(db, session_id, role="assistant", content=f"a{index}")
            ids.append(row.id)
        # A summary row must never become the head: appending afterwards still
        # links to the last *normal* row, not to the summary.
        summary = Message(
            session_id=session_id,
            role="user",
            kind="summary",
            content="earlier facts",
            covers_until=ids[-1],
            source_hash="a" * 64,
            summary_version=SUMMARY_VERSION,
        )
        db.add(summary)
        await db.flush()
        summary_id = summary.id
        tail = await append_message(db, session_id, role="assistant", content="a3")
        ids.append(tail.id)
        await db.commit()

    rows = await read_rows(session_id)
    by_id = {row.id: row for row in rows}
    # parent chain exactly in insertion (id) order
    assert len(ids) == 5
    assert ids[0] == rows[0].id
    assert by_id[ids[0]].parent_id is None
    for previous, current in zip(ids, ids[1:], strict=False):
        assert by_id[current].parent_id == previous, "parent chain must follow the id order"
    # the summary is present, has no parent, is not part of the active chain
    summary_row = by_id[summary_id]
    assert summary_row.kind == "summary"
    assert summary_row.parent_id is None
    # the head is the last normal row, not the summary
    assert by_id[ids[-1]].parent_id == ids[-2]


# --------------------------------------------------------------------------- #
# 2. Concurrent writers produce a single linear chain
# --------------------------------------------------------------------------- #


async def test_concurrent_writers_produce_single_chain(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        await append_message(db, session_id, role="user", content="seed")
        await db.commit()

    async def writer(tag: str) -> Message:
        async with session_factory() as db:
            message = await append_message(db, session_id, role="assistant", content=tag)
            await db.commit()
            return message

    results = await asyncio.gather(writer("w1"), writer("w2"))
    writer_ids = {message.id for message in results}
    assert len(writer_ids) == 2

    chain = await assert_chain_invariant(session_id)
    chain_ids = {row.id for row in chain}
    # seed + both writers survive; none lost
    assert len(chain) == 3
    assert writer_ids <= chain_ids
    parents = [row.parent_id for row in chain]
    # no fork: every active row has a distinct parent (one root + two children)
    assert len(set(parents)) == 3
    assert parents[0] is None


# --------------------------------------------------------------------------- #
# 3. edit_message semantics
# --------------------------------------------------------------------------- #


async def _build_edit_tree(session_id: str) -> tuple[int, int, int]:
    async with session_factory() as db:
        edit_call_id = uuid.uuid4().hex
        u1 = await append_message(db, session_id, role="user", content="u1")
        await append_message(db, session_id, role="assistant", content="a1")
        db.add(
            ToolCall(
                id=edit_call_id,
                session_id=session_id,
                name="shell",
                arguments={},
                risk_level="low",
                status=ToolCallStatus.pending.value,
            )
        )
        db.add(Approval(tool_call_id=edit_call_id, status=ApprovalStatus.pending.value))
        await append_message(db, session_id, role="user", content="u2")
        a2 = await append_message(db, session_id, role="assistant", content="a2")
        await db.commit()
        gen0 = int((await db.get(Session, session_id)).run_generation)
        return u1.id, a2.id, gen0, edit_call_id


async def test_edit_message_hides_tail_and_cancels_pending_calls(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path)
    u1_id, a2_id, gen0, edit_call_id = await _build_edit_tree(session_id)

    new_id = 0
    async with session_factory() as db:
        new = await edit_message(db, session_id, u1_id, "u1-new")
        await db.commit()
        new_id = new.id
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.status == SessionStatus.pending.value
        assert session.run_generation == gen0 + 1

    rows = await read_rows(session_id)
    by_id = {row.id: row for row in rows}
    # new user row: same parent as the old one (root -> None), new id
    new_row = by_id[new_id]
    assert new_row.role == "user"
    assert new_row.content == "u1-new"
    assert new_row.parent_id is None
    assert new_row.skipped is False
    # the old message and its whole live tail are hidden
    for message_id in (u1_id, a2_id):
        assert by_id[message_id].skipped is True
        # u2/a2 were inserted between, all part of the hidden tail
    assert all(row.skipped for row in rows if row.kind == "normal" and row.id != new_id)
    # the session row carries the new generation
    async with session_factory() as db:
        tc = await db.get(ToolCall, edit_call_id)
        approval = await db.scalar(select(Approval).where(Approval.tool_call_id == edit_call_id))
    assert tc is not None and tc.status == ToolCallStatus.cancelled.value
    assert approval is not None and approval.status == ApprovalStatus.cancelled.value

    # a second edit while the session is pending is refused with 409
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await edit_message(db, session_id, u1_id, "again")
        assert exc.value.status == 409
        await db.rollback()


# --------------------------------------------------------------------------- #
# 4. regenerate semantics
# --------------------------------------------------------------------------- #


_MODELS_WITH_EFFORT = """providers:
  cloud:
    kind: openai
    base_url: https://example.com/v1
models:
  basic:
    provider: cloud
    model: remote-basic
    context_window: 16000
    max_tokens: 2000
  smart:
    provider: cloud
    model: remote-smart
    context_window: 16000
    max_tokens: 2000
    reasoning:
      supported: true
      allowed_efforts: [low, medium, high]
      default_effort: low
      wire_parameter: reasoning_effort
routing:
  default: basic
  fallback: []
"""


async def test_regenerate_restarts_user_turn_and_applies_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_file = tmp_path / "models.yaml"
    models_file.write_text(_MODELS_WITH_EFFORT, encoding="utf-8")
    monkeypatch.setattr(get_settings(), "models_file", models_file, raising=False)
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        u1 = await append_message(
            db, session_id, role="user", content="u1", reasoning_effort="high"
        )
        a1 = await append_message(db, session_id, role="assistant", content="a1")
        await db.commit()
        u1_id, a1_id = u1.id, a1.id

    # regenerating a user message is refused
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await regenerate(db, session_id, u1_id)
        assert exc.value.status == 409
        await db.rollback()

    new_id = 0
    async with session_factory() as db:
        new_user = await regenerate(
            db, session_id, a1_id, llm_profile="smart", reasoning_effort="medium"
        )
        await db.commit()
        new_id = new_user.id
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.status == SessionStatus.pending.value
        assert session.llm_profile == "smart"

    rows = await read_rows(session_id)
    by_id = {row.id: row for row in rows}
    new_row = by_id[new_id]
    # copy of the user parent with the same parent (root) and the override effort
    assert new_row.role == "user"
    assert new_row.content == "u1"
    assert new_row.parent_id is None
    assert new_row.reasoning_effort == "medium"
    # the old user and the assistant answer are hidden
    assert by_id[u1_id].skipped is True
    assert by_id[a1_id].skipped is True


async def test_regenerate_without_override_keeps_user_effort(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        await append_message(db, session_id, role="user", content="u1", reasoning_effort="high")
        a1 = await append_message(db, session_id, role="assistant", content="a1")
        await db.commit()
        a1_id = a1.id
    async with session_factory() as db:
        new_user = await regenerate(db, session_id, a1_id)
        await db.commit()
        new_id = new_user.id
    by_id = {row.id: row for row in await read_rows(session_id)}
    assert by_id[new_id].reasoning_effort == "high"


# --------------------------------------------------------------------------- #
# 5. switch_branch back and forth
# --------------------------------------------------------------------------- #


async def test_switch_branch_back_and_forth_keeps_one_active_chain(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        u1 = await append_message(db, session_id, role="user", content="u1")
        a1 = await append_message(db, session_id, role="assistant", content="a1")
        u2 = await append_message(db, session_id, role="user", content="u2")
        a2 = await append_message(db, session_id, role="assistant", content="a2")
        db.add(
            Message(
                session_id=session_id,
                role="user",
                kind="summary",
                content="sum",
                covers_until=a2.id,
                source_hash="b" * 64,
                summary_version=SUMMARY_VERSION,
            )
        )
        await db.commit()
        u1_id, a1_id, u2_id, a2_id = u1.id, a1.id, u2.id, a2.id

    # branch 1 (from edit): u1 -> u1b ; then a1b appended on that branch
    async with session_factory() as db:
        u1b = await edit_message(db, session_id, u1_id, "u1b")
        await db.commit()
        u1b_id = u1b.id
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        session.status = SessionStatus.completed.value
        await db.commit()
    async with session_factory() as db:
        a1b = await append_message(db, session_id, role="assistant", content="a1b")
        await db.commit()
        a1b_id = a1b.id

    summary_id = 0
    expected_sets = (
        (a1b_id, (u1b_id, a1b_id)),  # branch 1 leaf
        (a2_id, (u1_id, a1_id, u2_id, a2_id)),  # branch 2 (original) leaf
        (a1b_id, (u1b_id, a1b_id)),
        (a2_id, (u1_id, a1_id, u2_id, a2_id)),
        (a1b_id, (u1b_id, a1b_id)),
    )
    for leaf, expected in expected_sets:
        async with session_factory() as db:
            path = await switch_branch(db, session_id, leaf)
            await db.commit()
            summary_id = int(
                await db.scalar(
                    select(Message.id).where(
                        Message.session_id == session_id, Message.kind == "summary"
                    )
                )
                or 0
            )
        assert [row.id for row in path] == list(expected)
        chain = await assert_chain_invariant(session_id)
        assert [row.id for row in chain] == list(expected)

        # summary rows are never touched by the branch switch
        rows = await read_rows(session_id)
        summary_row = next(row for row in rows if row.id == summary_id)
        assert summary_row.kind == "summary"
        assert summary_row.skipped is False
        assert summary_row.covers_until == a2_id

    # after the last switch the other branch is fully hidden
    rows = await read_rows(session_id)
    active_ids = {row.id for row in rows if row.kind == "normal" and not row.skipped}
    assert active_ids == {u1b_id, a1b_id}


# --------------------------------------------------------------------------- #
# 6. Races: running executor, Stop, stale fence
# --------------------------------------------------------------------------- #


async def test_race_edit_running_stops_and_stale_fence(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        u1 = await append_message(db, session_id, role="user", content="u1")
        a1 = await append_message(db, session_id, role="assistant", content="a1")
        await db.commit()
        u1_id, a1_id = u1.id, a1.id
        gen0 = int((await db.get(Session, session_id)).run_generation)

    now = datetime.now(UTC)
    # a running executor holds a live lease
    async with session_factory() as db:
        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=SessionStatus.running.value)
        )
        await db.commit()
    fence: RunFence | None = None
    async with session_factory() as db:
        fence = await claim(db, session_id, lease_seconds=300, now=now)
    assert fence is not None
    assert fence.generation == gen0

    # edit while the executor owns a live lease -> 409, nothing changed
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await edit_message(db, session_id, u1_id, "while-running")
        assert exc.value.status == 409
        await db.rollback()
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.status == SessionStatus.running.value
        assert session.run_generation == gen0
        assert session.owner_token == fence.token

    # Stop: bumps the generation and marks the session stopped
    async with session_factory() as db:
        assert await stop_generation(db, session_id) is True
        session = await db.get(Session, session_id)
        assert session is not None
        session.status = SessionStatus.stopped.value
        await db.commit()
        gen_after_stop = session.run_generation
    assert gen_after_stop == gen0 + 1

    # the stale (pre-Stop) fence can no longer write
    async with session_factory() as db:
        with pytest.raises(FenceLostError):
            await fence_check(db, fence, now=now)
        await db.rollback()

    # an edit after Stop passes: stopped is an idle status
    new_id = 0
    async with session_factory() as db:
        new_user = await edit_message(db, session_id, u1_id, "edited-after-stop")
        await db.commit()
        new_id = new_user.id
        session = await db.get(Session, session_id)
        assert session is not None
        assert session.run_generation == gen_after_stop + 1
        assert session.status == SessionStatus.pending.value

    by_id = {row.id: row for row in await read_rows(session_id)}
    assert by_id[u1_id].skipped is True
    assert by_id[a1_id].skipped is True
    assert by_id[new_id].role == "user"
    assert by_id[new_id].parent_id is None
    assert by_id[new_id].content == "edited-after-stop"


# --------------------------------------------------------------------------- #
# 7. Damaged parent links
# --------------------------------------------------------------------------- #


async def test_path_to_rejects_cross_session_and_forward_parent(tmp_path: Path) -> None:
    session_a = await new_session(tmp_path)
    session_b = await new_session(tmp_path)
    async with session_factory() as db:
        u_a = await append_message(db, session_a, role="user", content="u")
        a_a = await append_message(db, session_a, role="assistant", content="a")
        m_b = await append_message(db, session_b, role="user", content="m")
        a2 = await append_message(db, session_a, role="assistant", content="a2")
        await db.commit()
        u_a_id, a_a_id, m_b_id, a2_id = u_a.id, a_a.id, m_b.id, a2.id

    # parent from another session
    async with session_factory() as db:
        await db.execute(update(Message).where(Message.id == a_a_id).values(parent_id=m_b_id))
        await db.commit()
        with pytest.raises(TreeError) as exc:
            await path_to(db, session_a, a_a_id)
        assert exc.value.status == 409
        await db.rollback()

    # parent with a larger id than the row itself
    async with session_factory() as db:
        await db.execute(update(Message).where(Message.id == a_a_id).values(parent_id=a2_id))
        await db.commit()
        assert a2_id > a_a_id > u_a_id
        with pytest.raises(TreeError) as exc:
            await path_to(db, session_a, a_a_id)
        assert exc.value.status == 409
        await db.rollback()


# --------------------------------------------------------------------------- #
# 8. fork: copy path with new ids, skip checkpoint and tool calls
# --------------------------------------------------------------------------- #


async def test_fork_copies_path_and_skips_checkpoint_and_toolcall(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        await append_message(db, session_id, role="user", content="u1", reasoning_effort="high")
        a1 = await append_message(
            db,
            session_id,
            role="assistant",
            content="a1",
            tool_calls=[
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        )
        fork_call_id = uuid.uuid4().hex
        db.add(
            ToolCall(
                id=fork_call_id,
                session_id=session_id,
                name="read_file",
                arguments={},
                risk_level="low",
                status=ToolCallStatus.pending.value,
            )
        )
        db.add(
            Message(
                session_id=session_id,
                role="user",
                kind="summary",
                content="sum",
                covers_until=a1.id,
                source_hash="d" * 64,
                summary_version=SUMMARY_VERSION,
            )
        )
        await db.commit()
        a1_id = a1.id

    new_sid = ""
    async with session_factory() as db:
        copy = await fork(db, session_id, a1_id, title="forked")
        await db.commit()
        new_sid = copy.id

    assert new_sid != session_id
    async with session_factory() as db:
        copy_session = await db.get(Session, new_sid)
        assert copy_session is not None
        assert copy_session.status == SessionStatus.completed.value
        assert copy_session.title == "forked"
        messages = list(
            await db.scalars(
                select(Message).where(Message.session_id == new_sid).order_by(Message.id)
            )
        )
        assert len(messages) == 2
        assert all(message.kind == "normal" for message in messages)
        clone_u, clone_a = messages
        # new ids, strictly greater than the source ids
        assert clone_u.id > a1_id and clone_a.id > clone_u.id
        # parent chain is rebuilt with the new ids
        assert clone_u.parent_id is None
        assert clone_a.parent_id == clone_u.id
        # content and effort copied
        assert clone_u.content == "u1"
        assert clone_u.reasoning_effort == "high"
        assert clone_a.content == "a1"
        # checkpoint summary row is not copied
        summary_count = await db.scalar(
            select(func.count(Message.id)).where(
                Message.session_id == new_sid, Message.kind == "summary"
            )
        )
        assert int(summary_count or 0) == 0
        # the tool call row is not copied
        call_count = await db.scalar(
            select(func.count(ToolCall.id)).where(ToolCall.session_id == new_sid)
        )
        assert int(call_count or 0) == 0


# --------------------------------------------------------------------------- #
# 9. Property: random operation sequences keep the invariants
# --------------------------------------------------------------------------- #

_SEEDS = (1, 2, 3, 4, 5)
_STEPS = 50
_OPS = ("append", "edit", "regenerate", "switch_branch")


@dataclass(frozen=True, slots=True)
class _SavedCheckpoint:
    """A checkpoint saved by the test, plus the previous checkpoint used to chain it."""

    checkpoint: SummaryCheckpoint
    previous: SummaryCheckpoint | None


def _project_items(rows: Sequence[Message]) -> list[dict[str, Any]]:
    """Build the wire projection items of the given (ordered) rows."""
    sources = [SourceMessage.from_row(row) for row in rows]
    projection = project(HistorySnapshot(tuple(sources), None, None))
    return projection.items


@pytest.mark.parametrize("seed", _SEEDS)
async def test_tree_property_random_operations(tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        await append_message(db, session_id, role="system", content="sys")
        await append_message(db, session_id, role="user", content="first")
        await append_message(db, session_id, role="assistant", content="reply")
        await db.commit()

    saved: dict[int, _SavedCheckpoint] = {}
    previous: SummaryCheckpoint | None = None

    for step in range(_STEPS):
        rows_before = await read_rows(session_id)
        active_before = [row for row in rows_before if row.kind == "normal" and not row.skipped]
        active_users = [row.id for row in active_before if row.role == "user"]
        active_assistants = [row.id for row in active_before if row.role == "assistant"]
        all_normal = [row.id for row in rows_before if row.kind == "normal"]

        op = rng.choice(_OPS)
        target: int | None = None
        if op == "edit" and active_users:
            target = rng.choice(active_users)
        elif op == "regenerate" and active_assistants:
            target = rng.choice(active_assistants)
        elif op == "switch_branch" and all_normal:
            target = rng.choice(all_normal)

        effective = op if (op == "append" or target is not None) else "append"

        async with session_factory() as db:
            # tree ops run only while the session is idle; force it directly
            await db.execute(
                update(Session)
                .where(Session.id == session_id)
                .values(status=SessionStatus.completed.value)
            )
            if effective == "append":
                await append_message(db, session_id, role="user", content=f"u{step}")
            else:
                assert target is not None
                if effective == "edit":
                    await edit_message(db, session_id, target, f"edit-{step}")
                elif effective == "regenerate":
                    await regenerate(db, session_id, target)
                else:
                    await switch_branch(db, session_id, target)
            await db.commit()

        # central invariant after the step: active rows are one increasing chain
        await assert_chain_invariant(session_id)

        rows_after = await read_rows(session_id)
        sources = [SourceMessage.from_row(row) for row in rows_after]
        non_summary_ids = {row.id for row in rows_after if row.kind != "summary"}
        # build_snapshot only ever selects an eligible checkpoint
        snapshot = build_snapshot(sources)
        if snapshot.checkpoint is not None:
            checkpoint = snapshot.checkpoint
            assert len(checkpoint.source_hash) == 64
            assert all(char in _HEX for char in checkpoint.source_hash)
            assert checkpoint.covers_until in non_summary_ids
            assert checkpoint.version == SUMMARY_VERSION

        active_after = [row for row in rows_after if row.kind == "normal" and not row.skipped]
        if not active_after:
            continue
        covers = active_after[-1].id
        # save a fresh checkpoint only when it advances the greatest coverage,
        # so build_snapshot reliably selects it for the hash verification.
        max_covered = max(saved) if saved else 0
        if covers > max_covered:
            prefix = [row for row in active_after if row.id <= covers]
            items = _project_items(prefix)
            checkpoint = SummaryCheckpoint(
                text=f"step summary {step}",
                covers_until=covers,
                source_hash=chain_hash(previous, items),
                version=SUMMARY_VERSION,
            )
            async with session_factory() as db:
                await save_checkpoint(db, session_id, checkpoint, None)
            saved[covers] = _SavedCheckpoint(checkpoint, previous)
            previous = SummaryCheckpoint(
                text=checkpoint.text,
                covers_until=covers,
                source_hash=checkpoint.source_hash,
                version=checkpoint.version,
            )

            rows_final = await read_rows(session_id)
            snapshot_final = build_snapshot([SourceMessage.from_row(row) for row in rows_final])
            # build_snapshot picks the newest/viable checkpoint and its hash chains
            assert snapshot_final.checkpoint is not None
            assert snapshot_final.checkpoint.covers_until == covers
            saved_row = saved[covers]
            assert snapshot_final.checkpoint.source_hash == chain_hash(
                saved_row.previous, _project_items(prefix)
            )
            # the saved hash is exactly what we stored
            assert saved_row.checkpoint.source_hash == snapshot_final.checkpoint.source_hash


# --------------------------------------------------------------------------- #
# Self-check: empty / whitespace / wrong-target edits and boundaries
# --------------------------------------------------------------------------- #


async def test_edit_empty_or_whitespace_content_rejected_without_state_change(
    tmp_path: Path,
) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        u1 = await append_message(db, session_id, role="user", content="u1")
        await db.commit()
        u1_id = u1.id
        gen0 = int((await db.get(Session, session_id)).run_generation)
    for bad in ("", "   ", "\t\n "):
        async with session_factory() as db:
            with pytest.raises(TreeError) as exc:
                await edit_message(db, session_id, u1_id, bad)
            assert exc.value.status == 400
            await db.rollback()
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        # no partial write: still idle, generation untouched, no new row
        assert session.status == SessionStatus.completed.value
        assert session.run_generation == gen0
    assert len(await read_rows(session_id)) == 1


async def test_edit_wrong_role_and_missing_message_statuses(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path)
    other = await new_session(tmp_path)
    async with session_factory() as db:
        u1 = await append_message(db, session_id, role="user", content="u1")
        a1 = await append_message(db, session_id, role="assistant", content="a1")
        foreign = await append_message(db, other, role="user", content="foreign")
        await db.commit()
        u1_id, a1_id, foreign_id = u1.id, a1.id, foreign.id

    # editing an assistant message -> 409 (only user rows are editable)
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await edit_message(db, session_id, a1_id, "no")
        assert exc.value.status == 409
        await db.rollback()

    # editing a message of another session -> 404
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await edit_message(db, session_id, foreign_id, "no")
        assert exc.value.status == 404
        await db.rollback()

    # editing a nonexistent message -> 404
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await edit_message(db, session_id, u1_id + 1000, "no")
        assert exc.value.status == 404
        await db.rollback()


async def test_edit_skipped_message_is_rejected(tmp_path: Path) -> None:
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        u1 = await append_message(db, session_id, role="user", content="u1")
        await append_message(db, session_id, role="assistant", content="a1")
        await db.commit()
        u1_id = u1.id
    # hide the whole chain (simulate a prior switch away from this branch)
    async with session_factory() as db:
        await db.execute(
            update(Message)
            .where(Message.session_id == session_id, Message.kind == "normal")
            .values(skipped=True)
        )
        await db.commit()
    async with session_factory() as db:
        with pytest.raises(TreeError) as exc:
            await edit_message(db, session_id, u1_id, "no")
        assert exc.value.status == 409
        await db.rollback()


async def _two_turn_session(tmp_path: Path, **lease: Any) -> tuple[str, int, int]:
    """Return a session with user + assistant rows and the given owner/lease values."""
    session_id = await new_session(tmp_path)
    async with session_factory() as db:
        user = await append_message(db, session_id, role="user", content="u1")
        assistant = await append_message(db, session_id, role="assistant", content="a1")
        await db.execute(update(Session).where(Session.id == session_id).values(**lease))
        await db.commit()
        return session_id, user.id, assistant.id


async def _state(session_id: str) -> tuple[int, int, int]:
    async with session_factory() as db:
        generation = await db.scalar(select(Session.run_generation).where(Session.id == session_id))
        sessions = await db.scalar(select(func.count()).select_from(Session))
        messages = await db.scalar(select(func.count()).select_from(Message))
        return int(generation or 0), int(sessions or 0), int(messages or 0)


@pytest.mark.parametrize("operation", ["edit", "regenerate", "switch"])
async def test_expired_lease_on_cached_session_allows_tree_change(
    tmp_path: Path, operation: str
) -> None:
    expired = datetime.now(UTC) - timedelta(minutes=5)
    session_id, user_id, assistant_id = await _two_turn_session(
        tmp_path, owner_token="stale-owner", lease_until=expired
    )
    async with session_factory() as db:
        cached = await db.get(Session, session_id)  # SQLite returns a naive datetime here
        assert cached is not None and cached.lease_until is not None
        if operation == "edit":
            await edit_message(db, session_id, user_id, "u2")
        elif operation == "regenerate":
            await regenerate(db, session_id, assistant_id)
        else:
            await switch_branch(db, session_id, assistant_id)
        await db.commit()
        assert cached.owner_token is None  # the cached row is refreshed after the claim
    generation, _sessions, _messages = await _state(session_id)
    assert generation == 1


@pytest.mark.parametrize("operation", ["edit", "fork"])
async def test_unknown_lease_keeps_session_busy(tmp_path: Path, operation: str) -> None:
    # An owner without a lease end is not proven gone: refuse and change nothing.
    session_id, user_id, assistant_id = await _two_turn_session(
        tmp_path, owner_token="unknown-owner", lease_until=None
    )
    before = await _state(session_id)
    async with session_factory() as db:
        with pytest.raises(TreeError) as error:
            if operation == "edit":
                await edit_message(db, session_id, user_id, "u2")
            else:
                await fork(db, session_id, assistant_id, title="copy")
        assert error.value.status == 409
        await db.rollback()
    assert await _state(session_id) == before


async def test_regenerate_validates_effort_against_fresh_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_file = tmp_path / "models.yaml"
    models_file.write_text(_MODELS_WITH_EFFORT, encoding="utf-8")
    monkeypatch.setattr(get_settings(), "models_file", models_file, raising=False)
    session_id, _user_id, assistant_id = await _two_turn_session(tmp_path, llm_profile="smart")
    before = await _state(session_id)
    rows_before = [(row.id, row.skipped) for row in await read_rows(session_id)]
    async with session_factory() as db:
        cached = await db.get(Session, session_id)
        assert cached is not None and cached.llm_profile == "smart"
        # Another writer changes the profile to a model without reasoning control.
        async with session_factory() as other:
            await other.execute(
                update(Session).where(Session.id == session_id).values(llm_profile="basic")
            )
            await other.commit()
        with pytest.raises(ValueError):
            await regenerate(db, session_id, assistant_id, reasoning_effort="medium")
        await db.rollback()
    assert await _state(session_id) == before
    assert [(row.id, row.skipped) for row in await read_rows(session_id)] == rows_before
    async with session_factory() as db:
        status = await db.scalar(select(Session.status).where(Session.id == session_id))
    assert status == SessionStatus.completed.value
