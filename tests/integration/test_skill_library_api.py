"""Shared and project skills through the API, and the skills that sessions get."""

from __future__ import annotations

import io
import stat
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent import api
from agent.config import Settings, get_settings
from agent.database import session_factory
from agent.main import app
from agent.models import Message, Project
from agent.skill_library import project_skill_set
from agent.tools.base import ToolContext
from agent.tools.skill_tools import ReadSkillTool
from tests.integration.helpers import authed_client


def skill_md(name: str, description: str, body: str, extra: str = "") -> str:
    return f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n"


def write_skill(directory: Path, name: str, description: str, body: str) -> None:
    (directory / name).mkdir(parents=True, exist_ok=True)
    (directory / name / "SKILL.md").write_text(skill_md(name, description, body), "utf-8")


@pytest.fixture
def shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point both the API and session creation to a fresh shared skills directory."""
    directory = tmp_path / "shared-skills"
    monkeypatch.setattr(api.settings, "skills_dir", directory)
    app.dependency_overrides[get_settings] = lambda: api.settings
    try:
        yield directory
    finally:
        app.dependency_overrides.pop(get_settings, None)


def new_project(client: TestClient, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    response = client.post("/api/projects", json={"name": root.name, "root_path": str(root)})
    assert response.status_code == 201, response.text
    return dict(response.json())


def listed(client: TestClient, project_id: str | None = None) -> dict[tuple[str, str], Any]:
    query = f"?project_id={project_id}" if project_id else ""
    body = client.get(f"/api/skills{query}").json()
    return {(item["scope"], item["name"]): item for item in body["skills"]}


def used(client: TestClient, project_id: str) -> dict[str, str]:
    """Return the skills that a session of the project gets: name -> instructions."""

    async def load() -> dict[str, str]:
        async with session_factory() as db:
            project = await db.get(Project, project_id)
            assert project is not None
            skills = await project_skill_set(db, api.settings, project)
            return {skill.name: skill.body for skill in skills.skills}

    return dict(client.portal.call(load))  # type: ignore[union-attr]


def test_shared_skill_serves_every_project_and_project_skill_only_its_own(
    tmp_path: Path, shared: Path
) -> None:
    write_skill(shared, "review", "Review code", "Shared review steps")
    with authed_client() as client:
        first = new_project(client, tmp_path / "first")
        second = new_project(client, tmp_path / "second")
        write_skill(tmp_path / "first/.agent/skills", "deploy", "Deploy it", "First deploy")

        assert used(client, first["id"]) == {
            "review": "Shared review steps",
            "deploy": "First deploy",
        }
        assert used(client, second["id"]) == {"review": "Shared review steps"}
        # Without a project the list has only the shared skills.
        assert set(listed(client)) == {("shared", "review")}


def test_project_skill_replaces_shared_skill_with_the_same_name(
    tmp_path: Path, shared: Path
) -> None:
    write_skill(shared, "review", "Review code", "Shared review steps")
    with authed_client() as client:
        project = new_project(client, tmp_path / "own")
        write_skill(tmp_path / "own/.agent/skills", "review", "Our review", "Our review steps")

        items = listed(client, project["id"])
        assert items[("shared", "review")]["replaced"] is True
        assert items[("project", "review")]["replaced"] is False
        assert used(client, project["id"]) == {"review": "Our review steps"}

        # Turning the own skill off gives the project the shared skill again.
        response = client.put(
            "/api/skills/project/review/switch",
            json={"project_id": project["id"], "enabled": False},
        )
        assert response.status_code == 200, response.text
        assert used(client, project["id"]) == {"review": "Shared review steps"}
        assert listed(client, project["id"])[("shared", "review")]["replaced"] is False


def test_shared_default_and_project_choice(tmp_path: Path, shared: Path) -> None:
    write_skill(shared, "review", "Review code", "Shared review steps")
    with authed_client() as client:
        first = new_project(client, tmp_path / "first")
        second = new_project(client, tmp_path / "second")

        off = client.put("/api/skills/shared/review/switch", json={"enabled": False})
        assert off.status_code == 200, off.text
        assert used(client, first["id"]) == {} and used(client, second["id"]) == {}

        on = client.put(
            "/api/skills/shared/review/switch",
            json={"project_id": first["id"], "enabled": True},
        )
        assert on.status_code == 200, on.text
        assert set(used(client, first["id"])) == {"review"}
        assert used(client, second["id"]) == {}
        item = listed(client, first["id"])[("shared", "review")]
        assert (item["enabled"], item["default_enabled"], item["project_choice"]) == (
            True,
            False,
            True,
        )

        # Removing the own choice returns the project to the default.
        reset = client.put(
            "/api/skills/shared/review/switch",
            json={"project_id": first["id"], "enabled": None},
        )
        assert reset.status_code == 200, reset.text
        assert used(client, first["id"]) == {}


def test_create_writes_skill_files_where_the_scope_says(tmp_path: Path, shared: Path) -> None:
    with authed_client() as client:
        project = new_project(client, tmp_path / "proj")
        common = {"description": "Write release notes", "instructions": "List the changes."}
        made = client.post("/api/skills", json={"scope": "shared", "name": "notes", **common})
        assert made.status_code == 201, made.text
        own = client.post(
            "/api/skills",
            json={"scope": "project", "project_id": project["id"], "name": "notes", **common},
        )
        assert own.status_code == 201, own.text

        assert (shared / "notes/SKILL.md").is_file()
        assert (tmp_path / "proj/.agent/skills/notes/SKILL.md").is_file()
        again = client.post("/api/skills", json={"scope": "shared", "name": "notes", **common})
        assert again.status_code == 409
        bad = client.post("/api/skills", json={"scope": "shared", "name": "Bad Name", **common})
        assert bad.status_code == 422
        lost = client.post("/api/skills", json={"scope": "project", "name": "x", **common})
        assert lost.status_code == 422


def test_session_prompt_lists_shared_and_project_skills(tmp_path: Path, shared: Path) -> None:
    write_skill(shared, "review", "Review code", "Shared review steps")
    with authed_client() as client:
        project = new_project(client, tmp_path / "proj")
        write_skill(tmp_path / "proj/.agent/skills", "deploy", "Deploy it", "Deploy steps")
        session = client.post(
            "/api/sessions",
            json={"project_id": project["id"], "prompt": "Привет", "mode": "ask"},
        ).json()

        async def system_prompt() -> str:
            async with session_factory() as db:
                from sqlalchemy import select

                return str(
                    await db.scalar(
                        select(Message.content).where(
                            Message.session_id == session["id"], Message.role == "system"
                        )
                    )
                )

        prompt = client.portal.call(system_prompt)  # type: ignore[union-attr]
        assert "- review: Review code" in prompt
        assert "- deploy: Deploy it" in prompt


def test_update_keeps_other_header_fields_and_refuses_a_stale_revision(
    tmp_path: Path, shared: Path
) -> None:
    (shared / "notes").mkdir(parents=True)
    (shared / "notes/SKILL.md").write_text(
        skill_md("notes", "Old", "Old body", extra="license: MIT\n"), "utf-8"
    )
    with authed_client() as client:
        detail = client.get("/api/skills/shared/notes").json()
        change = {"description": "New", "instructions": "New body"}
        saved = client.put(
            "/api/skills/shared/notes", json={**change, "revision": detail["revision"]}
        )
        assert saved.status_code == 200, saved.text
        text = (shared / "notes/SKILL.md").read_text("utf-8")
        assert "license: MIT" in text and "description: New" in text and "New body" in text

        stale = client.put(
            "/api/skills/shared/notes", json={**change, "revision": detail["revision"]}
        )
        assert stale.status_code == 409


def test_delete_removes_the_directory_and_its_switches(tmp_path: Path, shared: Path) -> None:
    write_skill(shared, "review", "Review code", "Steps")
    (shared / "review/checklist.md").write_text("- tests", "utf-8")
    with authed_client() as client:
        project = new_project(client, tmp_path / "proj")
        client.put(
            "/api/skills/shared/review/switch",
            json={"project_id": project["id"], "enabled": False},
        )
        revision = client.get("/api/skills/shared/review").json()["revision"]
        assert client.delete(f"/api/skills/shared/review?revision={'0' * 64}").status_code == 409
        gone = client.delete(f"/api/skills/shared/review?revision={revision}")
        assert gone.status_code == 200, gone.text
        assert not (shared / "review").exists()

        # A new skill with the old name starts without the old project choice.
        write_skill(shared, "review", "Review code", "New steps")
        assert used(client, project["id"]) == {"review": "New steps"}


def archive(entries: dict[str, str], links: dict[str, str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        for name, text in entries.items():
            target.writestr(name, text)
        for name, destination in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            target.writestr(info, destination)
    return buffer.getvalue()


def test_import_archive_installs_skill_with_its_files(tmp_path: Path, shared: Path) -> None:
    package = archive(
        {
            "pdf-tools/SKILL.md": skill_md("pdf-tools", "Work with PDF", "See reference.md"),
            "pdf-tools/reference.md": "Use the page numbers.",
            "__MACOSX/pdf-tools/._SKILL.md": "junk",
        }
    )
    with authed_client() as client:
        project = new_project(client, tmp_path / "proj")
        response = client.post(
            f"/api/skills/import?scope=project&project_id={project['id']}&filename=pdf.zip",
            content=package,
        )
        assert response.status_code == 201, response.text
        installed = tmp_path / "proj/.agent/skills/pdf-tools"
        assert (installed / "reference.md").read_text("utf-8") == "Use the page numbers."
        assert not any(path.name.startswith(".import-") for path in installed.parent.iterdir())

        async def read() -> tuple[dict[str, Any], dict[str, Any]]:
            async with session_factory() as db:
                row = await db.get(Project, project["id"])
                assert row is not None
                tool = ReadSkillTool(await project_skill_set(db, api.settings, row))
            context = ToolContext(session_id="s", project_root=tmp_path / "proj")
            main = await tool.execute(context, {"name": "pdf-tools"})
            extra = await tool.execute(context, {"name": "pdf-tools", "file": "reference.md"})
            return main, extra

        main, extra = client.portal.call(read)  # type: ignore[union-attr]
        assert main["files"] == ["reference.md"]
        assert extra["content"] == "Use the page numbers."

        again = client.post(
            f"/api/skills/import?scope=project&project_id={project['id']}&filename=pdf.zip",
            content=package,
        )
        assert again.status_code == 409


def test_import_single_skill_file(shared: Path) -> None:
    with authed_client() as client:
        response = client.post(
            "/api/skills/import?scope=shared&filename=SKILL.md",
            content=skill_md("notes", "Notes", "Write notes").encode(),
        )
        assert response.status_code == 201, response.text
        assert (shared / "notes/SKILL.md").is_file()


@pytest.mark.parametrize(
    ("package", "message"),
    [
        (archive({"../evil/SKILL.md": skill_md("evil", "x", "x")}), "Недопустимое имя"),
        (
            archive({"a/SKILL.md": skill_md("a", "x", "x")}, links={"a/secret": "/etc/passwd"}),
            "ссылка",
        ),
        (archive({"a/readme.md": "no skill"}), "нет файла SKILL.md"),
        (b"not a zip", "не zip-архив"),
    ],
)
def test_import_refuses_unsafe_or_wrong_packages(
    shared: Path, package: bytes, message: str
) -> None:
    with authed_client() as client:
        response = client.post("/api/skills/import?scope=shared&filename=x.zip", content=package)
        assert response.status_code == 422, response.text
        assert message in response.json()["detail"]
        assert not shared.exists() or not any(shared.iterdir())


def test_broken_skill_is_reported_and_others_stay(shared: Path) -> None:
    write_skill(shared, "good", "Good", "Steps")
    (shared / "broken").mkdir()
    (shared / "broken/SKILL.md").write_text("no header", "utf-8")
    with authed_client() as client:
        body = client.get("/api/skills").json()
        assert [item["name"] for item in body["skills"]] == ["good"]
        assert len(body["errors"]) == 1 and body["errors"][0]["scope"] == "shared"


def test_settings_without_skills_dir_use_the_data_directory(tmp_path: Path) -> None:
    assert Settings(data_dir=tmp_path).shared_skills_dir == tmp_path / "skills"
