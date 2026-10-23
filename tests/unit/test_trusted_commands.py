from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.trusted_commands import (
    trusted_local_command,
    trusted_program_path,
    trusted_remote_command,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("readme")
    (root / "src" / "app.py").write_text("print(1)")
    return root


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "ls src",
        "cat README.md",
        "grep -n print src/app.py",
        "head -20 README.md",
        "tail -n 5 README.md",
        "wc -l README.md",
        "du -sh .",
        "pwd",
        "stat README.md",
        "sha256sum README.md",
        "grep -n 'two words' README.md",
    ],
)
def test_plain_read_only_commands_are_trusted(project: Path, command: str) -> None:
    assert trusted_local_command(command, project) is (
        trusted_program_path(command.split()[0]) is not None
    )
    assert trusted_remote_command(command) is (command.split()[0] in {"ls", "pwd", "du", "stat"})


@pytest.mark.parametrize(
    "command",
    [
        "",
        "rm -rf src",
        # Programs that run other programs.
        "env rm -rf /",
        "command rm README.md",
        "awk 'BEGIN { print 1 }'",
        "find . -delete",
        "xargs rm",
        "man -P id ls",
        # Programs that can write files or change the system.
        "sort -o out.txt README.md",
        "uniq README.md out.txt",
        "date -s 2020-01-01",
        # Shell syntax: chaining, pipes, redirection, substitution, globs.
        "ls; rm -rf src",
        "ls /tmp;rm -rf src",
        "ls && rm README.md",
        "ls | sh",
        "cat README.md > copy.md",
        "cat < README.md",
        "echo $(id)",
        "ls `id`",
        "ls $HOME",
        "ls *.md",
        "ls ~",
        "ls\nrm README.md",
        "ls \\; rm",
        # A program given by a path can be any binary.
        "./ls",
        "/tmp/ls -la",
        # Recursive reading reaches secret files inside the project.
        "grep -r token .",
        "grep -R token .",
        "grep --recursive token .",
        # Unparseable quoting.
        'cat "README.md',
        "grep -f.env README.md",
        "grep -if.env README.md",
        "grep --file=.env README.md",
        "grep --rec token .",
        "grep --dereference-r token .",
        "grep --direc=recurse token .",
        "md5sum -c manifest.txt",
        "sha1sum --check manifest.txt",
        "sha256sum --check=manifest.txt",
        "wc --files0-from=list.txt",
        "du -L .",
        "ls -L src",
        "head -n README.md",
        "head --lines=.env README.md",
        "tail --retry README.md",
    ],
)
def test_other_commands_need_approval(project: Path, command: str) -> None:
    assert trusted_local_command(command, project) is False
    assert trusted_remote_command(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "cat .env",
        "cat config/.env.local",
        "cat .ssh/id_rsa",
        "cat id_ed25519",
        "cat server.pem",
        "cat tls.key",
        "cat .aws/credentials",
        "cat .git-credentials",
        "head .pgpass",
        "grep --file=.env x README.md",
        "cat .bash_history",
        "grep -f .env x README.md",
        # Files that keep tokens or passwords in plain text.
        "cat .envrc",
        "cat terraform.tfstate",
        "cat infra/prod.tfstate.backup",
        "cat prod.tfvars",
        "cat .config/gh/hosts.yml",
        "cat .git/config",
    ],
)
def test_secret_files_need_approval(project: Path, command: str) -> None:
    assert trusted_local_command(command, project) is False
    assert trusted_remote_command(command) is False


def test_local_commands_stay_inside_the_project(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (project / "link.txt").symlink_to(outside)

    assert trusted_local_command("cat /etc/hosts", project) is False
    assert trusted_local_command("ls ../", project) is False
    assert trusted_local_command("cat src/../../outside.txt", project) is False
    # A symbolic link inside the project that points outside it.
    assert trusted_local_command("cat link.txt", project) is False
    assert trusted_local_command("cat src/app.py", project) is True


def test_remote_commands_may_use_absolute_paths_but_not_secrets() -> None:
    assert trusted_remote_command("cat /etc/hostname") is False
    assert trusted_remote_command("tail -n 50 /var/log/syslog") is False
    assert trusted_remote_command("df -h /") is True
    assert trusted_remote_command("cat /etc/shadow") is False
    assert trusted_remote_command("cat /etc/sudoers") is False
    assert trusted_remote_command("cat /proc/self/environ") is False
    assert trusted_remote_command("ls /root/.ssh") is False
    assert trusted_remote_command("cat /home/app/.env") is False
    assert trusted_remote_command("cat /etc/ssh/ssh_host_ed25519_key") is False
    assert trusted_remote_command("tail /root/.psql_history") is False
    assert trusted_remote_command("cat /var/run/secrets/kubernetes.io/token") is False


@pytest.mark.parametrize("target", [".env", "secrets/token", "nested/.aws/credentials"])
def test_links_to_project_secrets_need_approval(project: Path, target: str) -> None:
    secret = project / target
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("synthetic")
    (project / "public.txt").symlink_to(secret)
    assert trusted_local_command("cat public.txt", project) is False


def test_paths_with_missing_components_still_need_approval(project: Path, tmp_path: Path) -> None:
    (project / "outside").symlink_to(tmp_path)
    assert trusted_local_command("cat outside/missing.txt", project) is False
    (project / "-file").symlink_to(tmp_path / "outside.txt")
    assert trusted_local_command("cat -- -file", project) is False


@pytest.mark.parametrize(
    "command", ["cat /dev/fd/3", "cat /sys/kernel/notes", "cat /etc/../proc/self/environ"]
)
def test_remote_special_files_need_approval(command: str) -> None:
    assert trusted_remote_command(command) is False


@pytest.mark.parametrize(
    "command", ["head -n20 README.md", "tail --lines=20 README.md", "cat -- README.md"]
)
def test_understood_options_stay_trusted(project: Path, command: str) -> None:
    assert trusted_local_command(command, project) is True
    assert trusted_remote_command(command) is False


@pytest.mark.parametrize(
    "program", ["cat", "grep", "head", "tail", "wc", "md5sum", "sha1sum", "sha256sum"]
)
def test_remote_content_reads_need_approval_even_with_public_names(program: str) -> None:
    assert trusted_remote_command(f"{program} public.txt") is False


def test_program_lookup_does_not_use_inherited_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/untrusted")
    resolved = trusted_program_path("cat")
    assert resolved is not None
    assert Path(resolved).parent in {Path(directory) for directory in os.defpath.split(os.pathsep)}
    assert trusted_program_path("python") is None


def test_unavailable_system_program_needs_approval(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent.trusted_commands.shutil.which", lambda *args, **kwargs: None)
    assert trusted_local_command("cat README.md", project) is False
