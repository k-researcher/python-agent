"""Shared and project skills: the skill set of a project and changes made from the UI.

Shared skills are in ``Settings.shared_skills_dir`` and serve every project. Project skills
are in ``<project>/.agent/skills`` and serve only that project; a project skill replaces a
shared skill with the same name. The files are the source of the skills; the database keeps
only the switches:

- a shared skill is on in all projects by default, and this default can be turned off;
- each project can turn any shared or project skill on or off for itself.

A switched off skill is left out before the merge, so turning off a project skill gives the
project the shared skill with the same name again.
"""

from __future__ import annotations

import hashlib
import io
import os
import secrets
import shutil
import stat
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.models import Project, ProjectSkillSetting, SharedSkillDefault
from agent.safe_fs import SafeProjectFS
from agent.skill_prompt import (
    SkillSet,
    load_scoped_skills,
    merge_skills,
    project_skills_dir,
    project_skills_safe,
)
from agent.skills import (
    MAX_SKILL_BYTES,
    SKILL_MD,
    Skill,
    SkillError,
    load_skill,
    parse_skill_text,
    validate_header,
    validate_name,
)
from agent.tools.base import ToolError

Scope = Literal["shared", "project"]

MAX_PACKAGE_BYTES = 5 * 1024 * 1024
MAX_PACKAGE_FILES = 200
MAX_UNPACKED_BYTES = 20 * 1024 * 1024
_IGNORED_ENTRIES = frozenset({".DS_Store", "Thumbs.db"})


class SkillConflictError(SkillError):
    """The skill exists already or changed after the client read it."""


@dataclass(frozen=True)
class SkillState:
    """One skill as a project sees it."""

    scope: Scope
    skill: Skill
    # The effective state in the project; without a project, the default of a shared skill.
    enabled: bool
    # The default of a shared skill for all projects; None for a project skill.
    default_enabled: bool | None
    # The own choice of the project; None when the project follows the default.
    project_choice: bool | None
    # An enabled project skill with the same name is used instead of this shared skill.
    replaced: bool


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[SkillState, ...]
    errors: tuple[tuple[Scope, str], ...]

    def find(self, scope: Scope, name: str) -> SkillState:
        for state in self.skills:
            if state.scope == scope and state.skill.name == name:
                return state
        raise SkillError(f"Unknown skill name: {name}")

    def skill_set(self) -> SkillSet:
        """Return the skills that the project uses."""
        shared = [s.skill for s in self.skills if s.scope == "shared" and s.enabled]
        project = [s.skill for s in self.skills if s.scope == "project" and s.enabled]
        return SkillSet(
            skills=merge_skills(shared, project),
            errors=tuple(message for _scope, message in self.errors),
        )


def revision(content: str) -> str:
    """Return the version tag of a SKILL.md text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def skill_text(skill: Skill) -> str:
    return (skill.root / SKILL_MD).read_text(encoding="utf-8")


async def load_catalog(
    db: AsyncSession, settings: Settings, project: Project | None
) -> SkillCatalog:
    """Load the shared skills and, with a project, its own skills and switches."""
    shared, _, shared_errors = load_scoped_skills(None, settings.shared_skills_dir)
    errors: list[tuple[Scope, str]] = [("shared", message) for message in shared_errors]
    own: tuple[Skill, ...] = ()
    if project is not None:
        _, own, own_errors = load_scoped_skills(Path(project.root_path), None)
        errors.extend(("project", message) for message in own_errors)
    defaults = {row.name: row.enabled for row in await db.scalars(select(SharedSkillDefault))}
    choices: dict[tuple[str, str], bool] = {}
    if project is not None:
        rows = await db.scalars(
            select(ProjectSkillSetting).where(ProjectSkillSetting.project_id == project.id)
        )
        choices = {(row.scope, row.name): row.enabled for row in rows}

    own_states = [
        SkillState(
            scope="project",
            skill=skill,
            enabled=choices.get(("project", skill.name), True),
            default_enabled=None,
            project_choice=choices.get(("project", skill.name)),
            replaced=False,
        )
        for skill in own
    ]
    used_own = {state.skill.name for state in own_states if state.enabled}
    shared_states = []
    for skill in shared:
        default = defaults.get(skill.name, True)
        choice = choices.get(("shared", skill.name))
        enabled = default if choice is None else choice
        shared_states.append(
            SkillState(
                scope="shared",
                skill=skill,
                enabled=enabled,
                default_enabled=default,
                project_choice=choice,
                replaced=enabled and skill.name in used_own,
            )
        )
    return SkillCatalog(skills=(*shared_states, *own_states), errors=tuple(errors))


async def project_skill_set(db: AsyncSession, settings: Settings, project: Project) -> SkillSet:
    """Return the skills that sessions of the project can use."""
    return (await load_catalog(db, settings, project)).skill_set()


async def set_switch(
    db: AsyncSession,
    scope: Scope,
    name: str,
    project: Project | None,
    enabled: bool | None,
) -> None:
    """Store a switch; None removes it, so the default applies again.

    Without a project the switch is the default of a shared skill.
    """
    validate_name(name)
    if project is None:
        if scope != "shared":
            raise SkillError("A project skill needs a project")
        row = await db.get(SharedSkillDefault, name)
        if enabled is None:
            if row is not None:
                await db.delete(row)
        elif row is None:
            db.add(SharedSkillDefault(name=name, enabled=enabled))
        else:
            row.enabled = enabled
        return
    choice = await db.get(ProjectSkillSetting, (project.id, scope, name))
    if enabled is None:
        if choice is not None:
            await db.delete(choice)
    elif choice is None:
        db.add(ProjectSkillSetting(project_id=project.id, scope=scope, name=name, enabled=enabled))
    else:
        choice.enabled = enabled


async def forget_switches(
    db: AsyncSession, scope: Scope, name: str, project: Project | None
) -> None:
    """Remove the switches of a deleted skill, so a new skill with its name starts clean."""
    query = delete(ProjectSkillSetting).where(
        ProjectSkillSetting.scope == scope, ProjectSkillSetting.name == name
    )
    if scope == "project":
        if project is None:
            raise SkillError("A project skill needs a project")
        query = query.where(ProjectSkillSetting.project_id == project.id)
    else:
        await db.execute(delete(SharedSkillDefault).where(SharedSkillDefault.name == name))
    await db.execute(query)


def render_skill(header: dict[str, object], body: str) -> str:
    """Write a SKILL.md text: the YAML header, then the instructions."""
    dumped = yaml.safe_dump(header, allow_unicode=True, sort_keys=False, width=1000)
    return f"---\n{dumped.rstrip()}\n---\n\n{body.strip()}\n"


def _checked_text(header: dict[str, object], body: str, label: Path) -> str:
    content = render_skill(header, body)
    if len(content.encode("utf-8")) > MAX_SKILL_BYTES:
        raise SkillError(f"{SKILL_MD} is larger than {MAX_SKILL_BYTES} bytes")
    data, _ = parse_skill_text(content, label)
    validate_header(data, label)
    return content


def _base(settings: Settings, scope: Scope, project: Project | None) -> tuple[Path, str]:
    """Return the root for safe file operations and the skills directory inside it."""
    if scope == "shared":
        settings.shared_skills_dir.mkdir(parents=True, exist_ok=True)
        return settings.shared_skills_dir, ""
    if project is None:
        raise SkillError("A project skill needs a project")
    root = Path(project.root_path)
    if not project_skills_safe(root):
        raise SkillError(f"Project skills directory is not safe: {project_skills_dir(root)}")
    return root, project_skills_dir(root).relative_to(root).as_posix() + "/"


def _fs_error(exc: ToolError) -> SkillError:
    if "conflict" in str(exc) or "already exists" in str(exc):
        return SkillConflictError("Навык изменился или уже есть. Обновите страницу.")
    return SkillError(str(exc))


def _skill_path(state: SkillState, base: Path) -> str:
    return (state.skill.root / SKILL_MD).relative_to(base.resolve()).as_posix()


def create_skill(
    settings: Settings,
    catalog: SkillCatalog,
    scope: Scope,
    project: Project | None,
    name: str,
    description: str,
    body: str,
) -> Skill:
    """Create a skill directory with one SKILL.md file."""
    validate_name(name)
    return _create(
        settings, catalog, scope, project, {"name": name, "description": description}, body
    )


def _check_free(catalog: SkillCatalog, scope: Scope, name: str) -> None:
    if any(s.scope == scope and s.skill.name == name for s in catalog.skills):
        raise SkillConflictError(f"Навык {name} уже есть")


def _create(
    settings: Settings,
    catalog: SkillCatalog,
    scope: Scope,
    project: Project | None,
    header: dict[str, object],
    body: str,
) -> Skill:
    base, prefix = _base(settings, scope, project)
    name = validate_header(header, base)[0]
    _check_free(catalog, scope, name)
    content = _checked_text(header, body, base / prefix / name)
    try:
        SafeProjectFS(base).create_text_exclusive(f"{prefix}{name}/{SKILL_MD}", content)
    except ToolError as exc:
        raise _fs_error(exc) from exc
    return load_skill(base / prefix / name)


def update_skill(
    settings: Settings,
    state: SkillState,
    project: Project | None,
    description: str,
    body: str,
    expected: str,
) -> Skill:
    """Replace the description and the instructions; keep the other header fields."""
    base, _ = _base(settings, state.scope, project)
    current = skill_text(state.skill)
    if revision(current) != expected:
        raise SkillConflictError("Навык изменился в другом месте. Обновите страницу.")
    header, _ = parse_skill_text(current, state.skill.root)
    content = _checked_text({**header, "description": description}, body, state.skill.root)
    try:
        SafeProjectFS(base).replace_if_unchanged(_skill_path(state, base), current, content)
    except ToolError as exc:
        raise _fs_error(exc) from exc
    return load_skill(state.skill.root)


def delete_skill(
    settings: Settings, state: SkillState, project: Project | None, expected: str
) -> None:
    """Remove the skill directory with all its files."""
    base, prefix = _base(settings, state.scope, project)
    if revision(skill_text(state.skill)) != expected:
        raise SkillConflictError("Навык изменился в другом месте. Обновите страницу.")
    directory = state.skill.root
    parent = (base / prefix).resolve()
    if directory.parent != parent or not stat.S_ISDIR(os.lstat(directory).st_mode):
        raise SkillError("Skill directory is not a plain directory in the skills directory")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise SkillError("This system cannot remove a directory safely")
    shutil.rmtree(directory)


def _package_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Check the archive entries; return the files of the skill with paths inside it.

    SKILL.md is at the top of the archive or in its only top directory.
    """
    files: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.filename.startswith("__MACOSX/") or path.name in _IGNORED_ENTRIES:
            continue
        if "\\" in info.filename or "\x00" in info.filename or path.is_absolute():
            raise SkillError(f"Недопустимое имя в архиве: {info.filename}")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise SkillError(f"Недопустимое имя в архиве: {info.filename}")
        if info.flag_bits & 0x1:
            raise SkillError("Архив зашифрован")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise SkillError(f"В архиве есть ссылка: {info.filename}")
        if not info.is_dir():
            files.append((info, path))
    if len(files) > MAX_PACKAGE_FILES:
        raise SkillError(f"В архиве больше {MAX_PACKAGE_FILES} файлов")
    names = {path for _info, path in files}
    if PurePosixPath(SKILL_MD) in names:
        prefix = ""
    else:
        tops = {path.parts[0] for path in names}
        top = next(iter(tops)) if len(tops) == 1 else ""
        if not top or PurePosixPath(top, SKILL_MD) not in names:
            raise SkillError("В архиве нет файла SKILL.md")
        prefix = top
    return [(info, path.relative_to(prefix) if prefix else path) for info, path in files]


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: int) -> bytes:
    # The size in the archive header can lie, so the read itself is bounded.
    with archive.open(info) as source:
        data = source.read(budget + 1)
    if len(data) > budget:
        raise SkillError(f"Распакованный архив больше {MAX_UNPACKED_BYTES // (1024 * 1024)} МБ")
    return data


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as target:
        target.write(data)


def _plain_directories(base: Path, relative: str) -> Path:
    """Create the directories of ``relative`` under ``base``; refuse a link on the way."""
    current = base
    for part in PurePosixPath(relative).parts:
        current = current / part
        with suppress(FileExistsError):
            os.mkdir(current)
        if not stat.S_ISDIR(os.lstat(current).st_mode):
            raise SkillError(f"Skills directory is not a plain directory: {current}")
    return current


def import_skill(
    settings: Settings,
    catalog: SkillCatalog,
    scope: Scope,
    project: Project | None,
    filename: str,
    data: bytes,
) -> Skill:
    """Install a skill from a SKILL.md file or from a zip archive with a skill directory."""
    if len(data) > MAX_PACKAGE_BYTES:
        raise SkillError(f"Файл больше {MAX_PACKAGE_BYTES // (1024 * 1024)} МБ")
    if filename.lower().endswith(".md"):
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(f"{SKILL_MD} is not UTF-8 text") from exc
        header, body = parse_skill_text(content, Path(filename))
        return _create(settings, catalog, scope, project, header, body)

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise SkillError("Это не zip-архив и не файл SKILL.md") from exc
    with archive:
        members = _package_members(archive)
        manifest = next(info for info, path in members if path == PurePosixPath(SKILL_MD))
        raw = _read_member(archive, manifest, MAX_SKILL_BYTES)
        try:
            header, _body = parse_skill_text(raw.decode("utf-8"), Path(filename))
        except UnicodeDecodeError as exc:
            raise SkillError(f"{SKILL_MD} is not UTF-8 text") from exc
        name = validate_header(header, Path(filename))[0]
        _check_free(catalog, scope, name)
        base, inner = _base(settings, scope, project)
        directory = _plain_directories(base, inner)
        target = directory / name
        if target.exists() or target.is_symlink():
            raise SkillConflictError(f"Папка {name} уже есть")
        # Unpack next to the target under a hidden name, then move it in one step.
        staging = directory / f".import-{secrets.token_hex(8)}"
        staging.mkdir(mode=0o755)
        try:
            budget = MAX_UNPACKED_BYTES
            for info, path in members:
                unpacked = _read_member(archive, info, budget)
                budget -= len(unpacked)
                (staging / path).parent.mkdir(parents=True, exist_ok=True)
                _write_new(staging / path, unpacked)
            load_skill(staging)
            os.rename(staging, target)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise SkillError(f"Не удалось распаковать архив: {exc.strerror or exc}") from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return load_skill(target)
