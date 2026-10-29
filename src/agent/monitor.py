"""Session monitor: collect rows from the database and render a text table."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.models import Message, Project, Session, SessionStatus, ToolCall

_WORKING_STATUSES = {SessionStatus.running.value, SessionStatus.pending.value}
_AWAITING_STATUSES = {SessionStatus.awaiting_confirmation.value}
_ERROR_STATUSES = {SessionStatus.error.value}
# The remaining statuses (completed, stopped, interrupted) form the last group.

_REPLICA_ROLES = {"user", "assistant"}


@dataclass(frozen=True, slots=True)
class SessionRow:
    """One session row shown by the monitor table."""

    session_id: str
    role: str  # "main" or "child"
    name: str
    model: str
    mode: str
    status: str
    duration_seconds: int
    idle_seconds: int
    last_action: str
    project: str = ""
    path: str = ""


def format_duration(seconds: int | float) -> str:
    """Format a duration compactly: 45с, 12м, 1ч05."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}с"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes}м"
    hours = minutes // 60
    rest = minutes % 60
    return f"{hours}ч{rest:02d}"


def _aware(value: datetime) -> datetime:
    """Ensure a datetime carries UTC; SQLite returns naive values."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _char_width(char: str) -> int:
    """Return the terminal width of one character (2 for East Asian wide glyphs)."""
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Return the terminal display width of a string."""
    return sum(_char_width(char) for char in text)


def _fit(text: str, width: int) -> str:
    """Trim text to fit the display width, appending an ellipsis when needed."""
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    result = ""
    used = 0
    for char in text:
        step = _char_width(char)
        if used + step > width - 1:
            break
        result += char
        used += step
    return result + "…"


def _pad(text: str, width: int) -> str:
    """Left-align text in a column, padding with spaces to the given width."""
    return text + " " * (width - display_width(text))


# Column widths. Only the last column (action) adapts to the terminal width.
_ROLE_W = 4
_NAME_W = 12
_PROJECT_W = 12
_PATH_W = 22
_MODEL_W = 8
_MODE_W = 6
_STATUS_W = 9
_DURATION_W = 6
_IDLE_W = 6
_FIXED_WIDTHS = [
    _ROLE_W,
    _NAME_W,
    _PROJECT_W,
    _PATH_W,
    _MODEL_W,
    _MODE_W,
    _STATUS_W,
    _DURATION_W,
    _IDLE_W,
]
_FIXED_W = sum(_FIXED_WIDTHS)
_SEPARATOR = "│"

_ROLE_LABELS = {"main": "осн", "child": "дет"}

_STATUS_LABELS = {
    SessionStatus.running.value: "работает",
    SessionStatus.pending.value: "в очереди",
    SessionStatus.awaiting_confirmation.value: "ждёт",
    SessionStatus.completed.value: "готово",
    SessionStatus.stopped.value: "стоп",
    SessionStatus.error.value: "ошибка",
    SessionStatus.interrupted.value: "прервано",
}

_HEADER = [
    "Роль",
    "Имя",
    "Проект",
    "Путь",
    "Модель",
    "Режим",
    "Статус",
    "Идёт",
    "Тишина",
    "Действие",
]


def short_path(path: str, home: str | None = None) -> str:
    """Show a path with the home directory as ~."""
    base = home if home is not None else str(Path.home())
    if base and (path == base or path.startswith(base + "/")):
        return "~" + path[len(base) :]
    return path


def _row_values(row: SessionRow) -> list[str]:
    """Convert a session row into the string values for each table column."""
    return [
        _ROLE_LABELS.get(row.role, row.role[:3]),
        row.name,
        row.project,
        row.path,
        row.model,
        row.mode,
        _STATUS_LABELS.get(row.status, row.status),
        format_duration(row.duration_seconds),
        format_duration(row.idle_seconds),
        row.last_action,
    ]


def _render_line(values: list[str], action_width: int) -> str:
    """Render one data line with fixed columns and a flexible last one."""
    widths = _column_widths(action_width + _FIXED_W)
    count = len(widths)
    fixed = [_fit(value, width) for value, width in zip(values[:count], widths, strict=True)]
    fixed_line = _SEPARATOR.join(
        _pad(value, width) for value, width in zip(fixed, widths, strict=True)
    )
    action = max(0, action_width + _FIXED_W - sum(widths))
    return fixed_line + _SEPARATOR + _fit(values[count], action)


# Columns that shrink first on a narrow terminal, with their minimum widths.
_SHRINKABLE = {3: 6, 2: 6, 1: 6}  # path, project, name
_MIN_ACTION_W = 10


def _column_widths(budget: int) -> list[int]:
    """Return fixed column widths that leave room for the action column within budget."""
    widths = list(_FIXED_WIDTHS)
    excess = sum(widths) + _MIN_ACTION_W - budget
    for index, minimum in _SHRINKABLE.items():
        if excess <= 0:
            break
        cut = min(excess, widths[index] - minimum)
        widths[index] -= cut
        excess -= cut
    return widths


def render_table(rows: list[SessionRow], width: int) -> list[str]:
    """Render the monitor table; each line never exceeds ``width``.

    The last column is flexible and is trimmed to fill the remaining space.
    An empty list yields a single line with a notice.
    """
    if not rows:
        return [_fit("нет активных сессий", width)]
    separator_count = len(_HEADER) - 1
    action_width = width - _FIXED_W - separator_count
    header = _fit(_render_line(_HEADER, action_width), width)
    lines = [header]
    for row in rows:
        lines.append(_fit(_render_line(_row_values(row), action_width), width))
    return lines


async def _latest_tool(db: AsyncSession, session_id: str) -> ToolCall | None:
    """Return the newest tool call of a session by its update time."""
    result = await db.scalars(
        select(ToolCall)
        .where(ToolCall.session_id == session_id)
        .order_by(ToolCall.updated_at.desc(), ToolCall.id.desc())
        .limit(1)
    )
    return result.one_or_none()


async def _latest_replica(db: AsyncSession, session_id: str) -> Message | None:
    """Return the newest user or assistant message of a session."""
    result = await db.scalars(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.kind == "normal",
            Message.skipped.is_(False),
            Message.role.in_(_REPLICA_ROLES),
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(1)
    )
    return result.one_or_none()


def _describe_action(tool: ToolCall | None, replica: Message | None) -> tuple[str, datetime]:
    """Return the newest activity label and its timestamp."""
    if tool is not None and (
        replica is None or _aware(tool.updated_at) > _aware(replica.created_at)
    ):
        return f"tool:{tool.name}", _aware(tool.updated_at)
    if replica is not None:
        content = (replica.content or "").replace("\n", " ").strip()
        return f"{replica.role}: {content}", _aware(replica.created_at)
    return "—", _aware(tool.updated_at) if tool is not None else datetime.min.replace(tzinfo=UTC)


def _status_group(status: str) -> int:
    """Return the sort group of a status: working, awaiting, error, then the rest."""
    if status in _WORKING_STATUSES:
        return 0
    if status in _AWAITING_STATUSES:
        return 1
    if status in _ERROR_STATUSES:
        return 2
    return 3


async def collect(
    db: AsyncSession,
    *,
    limit: int = 50,
    history_minutes: int = 30,
    now: datetime | None = None,
) -> list[SessionRow]:
    """Return monitor rows for active and recently finished sessions.

    Rows sort by status group (working, awaiting confirmation, error, other)
    and, within a group, by the newest activity. ``limit`` bounds the result.
    """
    current = now or datetime.now(UTC)
    cutoff = current - timedelta(minutes=history_minutes)
    sessions = list(
        await db.scalars(
            select(Session).where(
                Session.archived.is_(False),
                or_(
                    Session.status.in_(_WORKING_STATUSES | _AWAITING_STATUSES),
                    Session.updated_at >= cutoff,
                ),
            )
        )
    )
    project_ids = {session.project_id for session in sessions}
    projects = {
        row.id: (row.name, row.root_path)
        for row in (
            await db.execute(
                select(Project.id, Project.name, Project.root_path).where(
                    Project.id.in_(project_ids)
                )
            )
        ).all()
    }
    entries: list[tuple[SessionRow, datetime]] = []
    for session in sessions:
        project_name, root_path = projects.get(session.project_id, ("", ""))
        tool = await _latest_tool(db, session.id)
        replica = await _latest_replica(db, session.id)
        action, event_time = _describe_action(tool, replica)
        created_at = _aware(session.created_at)
        updated_at = _aware(session.updated_at)
        last_time = max(updated_at, event_time)
        entries.append(
            (
                SessionRow(
                    session_id=session.id,
                    role="main" if session.parent_id is None else "child",
                    name=(session.title or "")[:40],
                    model=session.llm_profile or session.engine_profile or "",
                    mode=session.mode,
                    status=session.status,
                    duration_seconds=max(0, int((current - created_at).total_seconds())),
                    idle_seconds=max(0, int((current - last_time).total_seconds())),
                    last_action=action[:60],
                    project=project_name,
                    path=short_path(root_path),
                ),
                last_time,
            )
        )
    entries.sort(key=lambda item: (_status_group(item[0].status), -item[1].timestamp()))
    return [row for row, _ in entries[:limit]]
