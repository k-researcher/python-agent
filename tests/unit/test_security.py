from pathlib import Path

import pytest

from agent.security import PathGuard, PathSecurityError


def test_resolves_path_inside_project(tmp_path: Path) -> None:
    source = tmp_path / "example.txt"
    source.write_text("ok", encoding="utf-8")

    resolved = PathGuard(tmp_path).resolve("example.txt")

    assert resolved == source


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(PathSecurityError):
        PathGuard(tmp_path).resolve("../outside.txt")


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-dir"
    outside.mkdir(exist_ok=True)
    target = outside / "private.txt"
    target.write_text("private", encoding="utf-8")
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathSecurityError):
        PathGuard(tmp_path).resolve("link/private.txt")


def test_allows_new_file_with_existing_parent(tmp_path: Path) -> None:
    resolved = PathGuard(tmp_path).resolve("new.txt", must_exist=False)

    assert resolved == tmp_path / "new.txt"


@pytest.mark.parametrize("name", [".env", ".env.production", ".npmrc", "id_ed25519", "server.key"])
def test_blocks_secret_bearing_files(tmp_path: Path, name: str) -> None:
    secret = tmp_path / name
    secret.write_text("secret", encoding="utf-8")

    with pytest.raises(PathSecurityError, match="secret-bearing"):
        PathGuard(tmp_path).resolve(name)


def test_allows_environment_template(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text("KEY=", encoding="utf-8")

    assert PathGuard(tmp_path).resolve(".env.example") == template


def test_rejects_dangling_symlink_pointing_outside(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    (project / "out.txt").symlink_to(outside)

    with pytest.raises(PathSecurityError):
        PathGuard(project).resolve("out.txt", must_exist=False)
    assert not outside.exists()
