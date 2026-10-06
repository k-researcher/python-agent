from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agent.auth import check_request
from agent.config import Settings, get_settings
from agent.database import get_db, session_factory
from agent.events import broker, emit_event
from agent.model_registry import effective_registry, select_model
from agent.models import (
    Approval,
    ApprovalStatus,
    Event,
    KnowledgeDocument,
    Message,
    OutboundAudit,
    Project,
    Session,
    SessionStatus,
    ToolCall,
)
from agent.reasoning import validate_effort
from agent.runtime import TERMINAL_STATUSES, AgentSupervisor
from agent.schemas import (
    ApprovalDecision,
    ApprovalRead,
    EventRead,
    KnowledgeDocumentRead,
    MessageCreate,
    MessageRead,
    OutboundAuditRead,
    ProjectCreate,
    ProjectRead,
    SessionCreate,
    SessionRead,
)
from agent.session_service import create_session_record

router = APIRouter(prefix="/api")
settings: Settings = get_settings()
supervisor = AgentSupervisor(settings)


@router.get("/health")
async def health() -> dict[str, object]:
    """Public liveness probe; infrastructure details live behind auth in /ready."""
    return {"status": "ok", "service": "python-agent"}


@router.get("/ready")
async def ready() -> dict[str, object]:
    try:
        infrastructure = await supervisor.infrastructure_status()
    except Exception as exc:
        raise HTTPException(503, f"Infrastructure is unavailable: {exc}") from exc
    if settings.execution_mode == "redis" and not infrastructure.get("workers"):
        raise HTTPException(503, "No active agent workers")
    return {"status": "ready", **infrastructure}


@router.get("/config/public")
async def public_config(db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    models = await effective_registry(db, settings)
    default_model = models.get(models.default_id)
    return {
        "llm_configured": default_model.configured,
        "llm_model": default_model.model,
        "knowledge_base_enabled": settings.knowledge_base_enabled,
        "network_tools_enabled": settings.allow_network_tools,
        "raw_llm_log": settings.raw_llm_log,
        "llm_destination": urlsplit(default_model.base_url).hostname or "",
        "network_allowlist": settings.network_allowlist,
        "database_connections": sorted(settings.database_connections),
        "ssh_connections": sorted(settings.ssh_connections),
        "context_window": default_model.context_window,
        "context_reserved_tokens": settings.context_reserved_tokens,
        "max_child_depth": settings.max_child_depth,
        "api_auth_enabled": True,
        "execution_mode": settings.execution_mode,
        "default_llm_profile": models.default_id,
        "models_source": models.source,
        "models_checksum": models.checksum,
        "llm_profiles": [
            {
                "name": model.id,
                "model": model.model,
                "provider": model.provider_id,
                "kind": model.kind,
                "destination": urlsplit(model.base_url).hostname or "",
                "configured": model.configured,
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "reasoning_efforts": model.reasoning.allowed_efforts,
                "default_reasoning_effort": model.reasoning.default_effort,
            }
            for model in models.models.values()
        ],
    }


@router.get("/projects", response_model=list[ProjectRead])
async def list_projects(db: AsyncSession = Depends(get_db)) -> list[Project]:
    return list((await db.execute(select(Project).order_by(Project.name))).scalars())


@router.post("/projects", response_model=ProjectRead, status_code=201)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)) -> Project:
    try:
        root = Path(payload.root_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise HTTPException(400, "Project directory does not exist") from exc
    if not root.is_dir():
        raise HTTPException(400, "Project root must be a directory")

    existing = (
        await db.execute(select(Project).where(Project.root_path == str(root)))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(409, "Project directory is already registered")

    project = Project(name=payload.name, root_path=str(root))
    db.add(project)
    await db.commit()
    await db.refresh(project)
    await emit_event(db, "project.created", {"project_id": project.id})
    return project


@router.get("/sessions", response_model=list[SessionRead])
async def list_sessions(
    archived: bool = Query(default=False), db: AsyncSession = Depends(get_db)
) -> list[Session]:
    query = select(Session).where(Session.archived == archived).order_by(Session.updated_at.desc())
    return list((await db.execute(query)).scalars())


@router.post("/sessions", response_model=SessionRead, status_code=201)
async def create_session(payload: SessionCreate, db: AsyncSession = Depends(get_db)) -> Session:
    project = await db.get(Project, payload.project_id)
    if project is None:
        raise HTTPException(404, "Project not found")

    try:
        session = await create_session_record(
            db,
            settings,
            project,
            payload.prompt,
            payload.mode,
            llm_profile=payload.llm_profile,
            title=payload.title,
            configuration={"depth": 0}
            | ({"reasoning_effort": payload.reasoning_effort} if payload.reasoning_effort else {}),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await emit_event(db, "session.created", {"status": session.status}, session.id)
    if payload.auto_start:
        await supervisor.start(session.id)
    return session


@router.get("/sessions/{session_id}", response_model=SessionRead)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)) -> Session:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


@router.get("/sessions/{session_id}/children", response_model=list[SessionRead])
async def list_children(session_id: str, db: AsyncSession = Depends(get_db)) -> list[Session]:
    if await db.get(Session, session_id) is None:
        raise HTTPException(404, "Session not found")
    query = select(Session).where(Session.parent_id == session_id).order_by(Session.created_at)
    return list((await db.execute(query)).scalars())


@router.get("/sessions/{session_id}/outbound-audit", response_model=list[OutboundAuditRead])
async def list_outbound_audit(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[OutboundAudit]:
    if await db.get(Session, session_id) is None:
        raise HTTPException(404, "Session not found")
    query = (
        select(OutboundAudit)
        .where(OutboundAudit.session_id == session_id)
        .order_by(OutboundAudit.id.desc())
        .limit(limit)
    )
    return list((await db.execute(query)).scalars())


@router.get("/sessions/{session_id}/context")
async def session_context(session_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, int]:
    session = (
        await db.execute(
            select(Session).where(Session.id == session_id).options(selectinload(Session.messages))
        )
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(404, "Session not found")
    model = select_model(await effective_registry(db, settings), session.llm_profile, strict=False)
    context = supervisor.context_for(model)
    result = context.prepare(session.messages)
    return {
        "approximate_tokens": result.approximate_tokens,
        "omitted_messages": result.omitted_messages,
        "truncated_tool_results": result.truncated_tool_results,
        "context_window": model.context_window,
        "reserved_tokens": model.context_window - context.budget,
    }


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def list_messages(session_id: str, db: AsyncSession = Depends(get_db)) -> list[Message]:
    if await db.get(Session, session_id) is None:
        raise HTTPException(404, "Session not found")
    query = select(Message).where(Message.session_id == session_id).order_by(Message.id)
    return list((await db.execute(query)).scalars())


@router.post("/sessions/{session_id}/messages", response_model=MessageRead, status_code=201)
async def add_message(
    session_id: str, payload: MessageCreate, db: AsyncSession = Depends(get_db)
) -> Message:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status in {SessionStatus.running.value, SessionStatus.awaiting_confirmation.value}:
        raise HTTPException(409, f"Cannot add a message while session is {session.status}")

    if payload.reasoning_effort is not None:
        model = select_model(
            await effective_registry(db, settings), session.llm_profile, strict=False
        )
        try:
            validate_effort(model, payload.reasoning_effort)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    message = Message(
        session_id=session.id,
        role="user",
        content=payload.content,
        reasoning_effort=payload.reasoning_effort,
    )
    db.add(message)
    session.status = SessionStatus.pending.value
    session.stop_requested = False
    session.error = None
    await db.commit()
    await db.refresh(message)
    await emit_event(db, "message.created", {"role": "user", "message_id": message.id}, session.id)
    await supervisor.start(session.id)
    return message


@router.post("/sessions/{session_id}/start", status_code=202)
async def start_session(session_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, bool]:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status == SessionStatus.awaiting_confirmation.value:
        raise HTTPException(409, "Resolve pending approvals before starting")
    if session.status in TERMINAL_STATUSES | {SessionStatus.interrupted.value}:
        session.status = SessionStatus.pending.value
        session.stop_requested = False
        session.error = None
        await db.commit()
    return {"started": await supervisor.start(session.id)}


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, bool]:
    if await db.get(Session, session_id) is None:
        raise HTTPException(404, "Session not found")
    await supervisor.stop(session_id)
    return {"stopped": True}


@router.post("/sessions/{session_id}/archive")
async def archive_session(session_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, bool]:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status not in TERMINAL_STATUSES | {SessionStatus.interrupted.value}:
        raise HTTPException(409, "Only inactive sessions can be archived")
    session.archived = True
    await db.commit()
    await emit_event(db, "session.archived", {}, session.id)
    return {"archived": True}


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)) -> None:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status not in TERMINAL_STATUSES | {SessionStatus.interrupted.value}:
        raise HTTPException(409, "Stop the session before deleting it")
    child_id = (
        await db.execute(select(Session.id).where(Session.parent_id == session_id).limit(1))
    ).scalar_one_or_none()
    if child_id is not None:
        raise HTTPException(409, "Delete or archive child sessions first")
    await db.execute(delete(Session).where(Session.id == session_id))
    await db.commit()
    await broker.broadcast({"type": "session.deleted", "session_id": session_id, "payload": {}})


@router.get("/sessions/{session_id}/approvals", response_model=list[ApprovalRead])
async def list_approvals(
    session_id: str, pending_only: bool = True, db: AsyncSession = Depends(get_db)
) -> list[ApprovalRead]:
    query = (
        select(Approval)
        .join(ToolCall)
        .where(ToolCall.session_id == session_id)
        .options(selectinload(Approval.tool_call))
        .order_by(Approval.created_at)
    )
    if pending_only:
        query = query.where(Approval.status == ApprovalStatus.pending.value)
    approvals = list((await db.execute(query)).scalars())
    return [
        ApprovalRead(
            id=item.id,
            tool_call_id=item.tool_call_id,
            tool_name=item.tool_call.name,
            arguments=item.tool_call.arguments,
            risk_level=item.tool_call.risk_level,
            status=item.status,
            comment=item.comment,
            created_at=item.created_at,
        )
        for item in approvals
    ]


@router.post("/approvals/{approval_id}")
async def resolve_approval(approval_id: str, payload: ApprovalDecision) -> dict[str, str]:
    try:
        session_id = await supervisor.resolve_approval(
            approval_id, payload.decision, payload.comment
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"session_id": session_id, "decision": payload.decision}


@router.get("/events", response_model=list[EventRead])
async def list_events(
    after_id: int = 0,
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[Event]:
    query = select(Event).where(Event.id > after_id).order_by(Event.id).limit(limit)
    return list((await db.execute(query)).scalars())


@router.get("/knowledge", response_model=list[KnowledgeDocumentRead])
async def list_knowledge(
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[KnowledgeDocument]:
    if not settings.knowledge_base_enabled:
        raise HTTPException(404, "Knowledge base is disabled")
    query = select(KnowledgeDocument).order_by(KnowledgeDocument.updated_at.desc()).limit(limit)
    return list((await db.execute(query)).scalars())


WS_AUTH_RECHECK_SECONDS = 5
WS_HEARTBEAT_SECONDS = 25


async def _websocket_allowed(websocket: WebSocket) -> bool:
    async with session_factory() as db:
        check = await check_request(
            db, settings, websocket.headers, websocket.cookies, "GET", require_origin=True
        )
    return check.ok


async def websocket_endpoint(websocket: WebSocket) -> None:
    # Host, Origin and the session cookie are verified before the handshake is accepted.
    if not await _websocket_allowed(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    await websocket.send_json({"type": "connected", "payload": {"service": "python-agent"}})
    loop = asyncio.get_running_loop()
    last_sent = last_checked = loop.time()
    try:
        async with broker.subscribe() as queue:
            while True:
                # Check the session on a clock, so a steady flow of events cannot skip it.
                if loop.time() - last_checked >= WS_AUTH_RECHECK_SECONDS:
                    if not await _websocket_allowed(websocket):
                        await websocket.close(code=4401, reason="Session expired or revoked")
                        return
                    last_checked = loop.time()
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=WS_AUTH_RECHECK_SECONDS)
                    await websocket.send_json(event)
                    last_sent = loop.time()
                except TimeoutError:
                    if loop.time() - last_sent >= WS_HEARTBEAT_SECONDS:
                        await websocket.send_json({"type": "heartbeat", "payload": {}})
                        last_sent = loop.time()
    except WebSocketDisconnect:
        return
