from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.models import Message, Project, Session, SessionStatus


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
    selected_profile = llm_profile or settings.default_llm_profile
    settings.resolve_llm_profile(selected_profile)
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
        configuration=configuration or {},
    )
    db.add(session)
    await db.flush()
    db.add_all(
        [
            Message(session_id=session.id, role="system", content=system_prompt),
            Message(session_id=session.id, role="user", content=prompt),
        ]
    )
    await db.commit()
    await db.refresh(session)
    return session
