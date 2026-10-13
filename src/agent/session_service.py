from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.message_tree import append_message
from agent.model_registry import effective_registry, select_model
from agent.models import Project, Session, SessionStatus
from agent.reasoning import validate_effort


async def create_session_record(
    db: AsyncSession,
    settings: Settings,
    project: Project,
    prompt: str,
    mode: str,
    *,
    llm_profile: str | None = None,
    title: str | None = None,
    parent_id: str | None = None,
    configuration: dict[str, Any] | None = None,
) -> Session:
    model = select_model(await effective_registry(db, settings), llm_profile, role=mode)
    selected_profile = model.id
    configuration = dict(configuration or {})
    validate_effort(model, configuration.get("reasoning_effort"))
    prompt_file = settings.prompts_dir / f"{mode}.md"
    if not prompt_file.is_file():
        raise ValueError(f"Prompt is missing for mode {mode}")

    system_prompt = prompt_file.read_text(encoding="utf-8")
    system_prompt += f"\n\nКорневая директория проекта: `{project.root_path}`"
    session = Session(
        project_id=project.id,
        parent_id=parent_id,
        mode=mode,
        llm_profile=selected_profile,
        title=title or prompt.strip().replace("\n", " ")[:120],
        status=SessionStatus.pending.value,
        configuration=configuration,
    )
    db.add(session)
    await db.flush()
    # Explicit parents: the system prompt is the root, the task is its child.
    await append_message(db, session.id, role="system", content=system_prompt)
    await append_message(db, session.id, role="user", content=prompt)
    await db.commit()
    await db.refresh(session)
    return session
