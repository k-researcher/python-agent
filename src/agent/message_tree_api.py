"""HTTP routes for message tree operations.

The routes mutate the message tree of a session: edit a user message, regenerate an
assistant turn, switch to a sibling branch, fork a session or inspect the sibling
versions of a message. Every path starts with ``/api`` so the same browser-auth
middleware that protects ``agent.api`` also protects these routes; the handlers only
declare the database dependency.

Each tree operation runs in one transaction and is committed before a new run is
scheduled, so an executor never observes a partial tree. On any error the transaction
is rolled back before the exception propagates. Edit and regenerate claim a new run
generation in that transaction and, after the commit, dispatch exactly that generation
with the same contract as sending a message: a refused or failed dispatch marks only
that pending generation as an error, keeps the saved rows and asks the client to retry
Start, not the tree operation.
"""

from __future__ import annotations

from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.api import _dispatch_session_run, _engine_access
from agent.database import get_db
from agent.message_tree import (
    TreeError,
    edit_message,
    fork,
    regenerate,
    siblings,
    switch_branch,
)
from agent.models import Message, Session
from agent.schemas import MessageRead, SessionRead

router = APIRouter(prefix="/api")


async def _claimed_generation(db: AsyncSession, session_id: str) -> int:
    """Read the run generation that the tree operation claimed, in its own transaction."""
    generation = await db.scalar(sa.select(Session.run_generation).where(Session.id == session_id))
    if generation is None:
        raise TreeError("Session not found", 404)
    return int(generation)


DbSession = Annotated[AsyncSession, Depends(get_db)]


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=2_000_000)


class RegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_profile: str | None = Field(default=None, min_length=1, max_length=100)
    reasoning_effort: str | None = Field(default=None, max_length=20)


class BranchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leaf_id: int


class ForkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=300)


def _http_error(exc: TreeError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


@router.post(
    "/sessions/{session_id}/messages/{message_id}/edit",
    response_model=MessageRead,
    status_code=201,
)
async def edit_message_endpoint(
    session_id: str,
    message_id: int,
    payload: EditRequest,
    db: DbSession,
) -> MessageRead:
    try:
        message = await edit_message(db, session_id, message_id, payload.content)
        generation = await _claimed_generation(db, session_id)
        response = MessageRead.model_validate(message)
        await db.commit()
    except TreeError as exc:
        await db.rollback()
        raise _http_error(exc) from exc
    except Exception:
        await db.rollback()
        raise
    await _dispatch_session_run(db, session_id, generation)
    return response


@router.post(
    "/sessions/{session_id}/messages/{message_id}/regenerate",
    response_model=MessageRead,
    status_code=201,
)
async def regenerate_endpoint(
    session_id: str,
    message_id: int,
    payload: RegenerateRequest,
    db: DbSession,
) -> MessageRead:
    try:
        message = await regenerate(
            db,
            session_id,
            message_id,
            llm_profile=payload.llm_profile,
            reasoning_effort=payload.reasoning_effort,
        )
        generation = await _claimed_generation(db, session_id)
        response = MessageRead.model_validate(message)
        await db.commit()
    except TreeError as exc:
        await db.rollback()
        raise _http_error(exc) from exc
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        await db.rollback()
        raise
    await _dispatch_session_run(db, session_id, generation)
    return response


@router.post("/sessions/{session_id}/branch", response_model=list[MessageRead])
async def branch_endpoint(
    session_id: str,
    payload: BranchRequest,
    db: DbSession,
) -> list[Message]:
    try:
        path = await switch_branch(db, session_id, payload.leaf_id)
        await db.commit()
    except TreeError as exc:
        await db.rollback()
        raise _http_error(exc) from exc
    except Exception:
        await db.rollback()
        raise
    for row in path:
        await db.refresh(row)
    return path


@router.post(
    "/sessions/{session_id}/messages/{message_id}/fork",
    response_model=SessionRead,
    status_code=201,
)
async def fork_endpoint(
    session_id: str,
    message_id: int,
    payload: ForkRequest,
    request: Request,
    db: DbSession,
) -> Session:
    try:
        source = await db.get(Session, session_id)
        if source is None:
            raise TreeError("Session not found", 404)
        await _engine_access(db, source, request)
        title = payload.title or f"{source.title} (ветка)"
        session = await fork(
            db,
            session_id,
            message_id,
            title=title,
            actor_user_id=getattr(request.state, "actor_user_id", None),
        )
        await db.commit()
        await db.refresh(session)
    except TreeError as exc:
        await db.rollback()
        raise _http_error(exc) from exc
    except Exception:
        await db.rollback()
        raise
    return session


@router.get("/sessions/{session_id}/messages/{message_id}/siblings")
async def siblings_endpoint(
    session_id: str,
    message_id: int,
    db: DbSession,
) -> dict[str, object]:
    try:
        result = await siblings(db, session_id, message_id)
    except TreeError as exc:
        await db.rollback()
        raise _http_error(exc) from exc
    except Exception:
        await db.rollback()
        raise
    return {"ids": result.ids, "active_index": result.active_index}
