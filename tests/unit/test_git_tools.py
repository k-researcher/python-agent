from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent.tools.base import ToolContext, ToolError
from agent.tools.git_tools import (
    MAX_COMMAND_OUTPUT,
    REVISION_PATTERN,
    GitDiffTool,
    GitLogTool,
    GitShowTool,
    GitStatusTool,
    _filter_diff_output,
    _run_git,
    _validate_paths,
    parse_status,
)


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git inside `repo` and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def write(repo: Path, name: str, content: str) -> Path:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def git_context(repo: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=repo)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a git repository with two commits and a clean working tree."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main", "-q"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=project,
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    write(project, "a.txt", "alpha\n")
    run_git(project, "add", "a.txt")
    run_git(project, "commit", "-m", "first commit", "--no-gpg-sign")
    write(project, "b.txt", "beta\n")
    write(project, "dir/c.txt", "gamma\n")
    run_git(project, "add", ".")
    run_git(project, "commit", "-m", "second commit", "--no-gpg-sign")
    return project


async def test_status_reports_modified_new_and_staged_files(repo: Path) -> None:
    write(repo, "a.txt", "alpha changed\n")
    write(repo, "untracked.py", "x = 1\n")
    write(repo, "staged.txt", "s\n")
    run_git(repo, "add", "staged.txt")

    result = await GitStatusTool().execute(git_context(repo), {})

    assert result["branch"] == "main"
    files = {(item["path"]): (item["index"], item["worktree"]) for item in result["files"]}
    assert files["a.txt"] == (" ", "M")
    assert files["staged.txt"] == ("A", " ")
    assert files["untracked.py"] == ("?", "?")


async def test_status_branch_parsing_without_upstream(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "master", "-q"], cwd=project, check=True)
    write(project, "f.txt", "content\n")
    subprocess.run(["git", "add", "f.txt"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "first", "--no-gpg-sign"], cwd=project, check=True)

    result = await GitStatusTool().execute(git_context(project), {})

    assert result["branch"] == "master"


async def test_status_empty_repo(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main", "-q"], cwd=project, check=True)

    result = await GitStatusTool().execute(git_context(project), {})

    assert result["branch"] == "main"
    assert result["files"] == []


async def test_status_detached_head(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main", "-q"], cwd=project, check=True)
    write(project, "f.txt", "content\n")
    subprocess.run(["git", "add", "f.txt"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "first", "--no-gpg-sign"], cwd=project, check=True)
    subprocess.run(["git", "checkout", "-q", "HEAD~0"], cwd=project, check=True)

    result = await GitStatusTool().execute(git_context(project), {})

    assert result["branch"] is None


def test_parse_status_entries(various_status_text: str) -> None:
    result = parse_status(various_status_text)

    assert result["branch"] == "feature"
    assert result["files"] == [
        {"path": "a.txt", "index": "M", "worktree": " ", "sensitive": False},
        {"path": "b.txt", "index": " ", "worktree": "M", "sensitive": False},
        {"path": "new.py", "index": "?", "worktree": "?", "sensitive": False},
        {"path": "renamed.py", "index": "R", "worktree": " ", "sensitive": False},
    ]


@pytest.fixture
def various_status_text() -> str:
    return (
        "## feature...origin/feature [ahead 2]\n"
        "M  a.txt\n"
        " M b.txt\n"
        "?? new.py\n"
        "R  old.py -> renamed.py\n"
    )


async def test_diff_working_tree_default(repo: Path) -> None:
    write(repo, "a.txt", "alpha v2\n")

    result = await GitDiffTool().execute(git_context(repo), {})

    assert "alpha v2" in result["diff"]
    assert "+alpha v2" in result["diff"]
    assert "-alpha" in result["diff"]
    assert result["truncated"] is False


async def test_diff_staged(repo: Path) -> None:
    write(repo, "a.txt", "alpha staged\n")
    run_git(repo, "add", "a.txt")

    unstaged = await GitDiffTool().execute(git_context(repo), {})
    staged = await GitDiffTool().execute(git_context(repo), {"staged": True})

    assert "alpha staged" not in unstaged["diff"]
    assert "+alpha staged" in staged["diff"]


async def test_diff_with_paths(repo: Path) -> None:
    write(repo, "a.txt", "changed a\n")
    write(repo, "b.txt", "changed b\n")

    result = await GitDiffTool().execute(git_context(repo), {"paths": ["a.txt"]})

    assert "changed a" in result["diff"]
    assert "changed b" not in result["diff"]
    assert "b.txt" not in result["diff"]


async def test_diff_context_lines_zero(repo: Path) -> None:
    write(repo, "dir/c.txt", "one\ntwo\nthree\nfour\n")
    run_git(repo, "add", "dir/c.txt")
    run_git(repo, "commit", "-m", "four lines", "--no-gpg-sign")
    write(repo, "dir/c.txt", "one\ntwo\nTHREE\nfour\n")

    result = await GitDiffTool().execute(git_context(repo), {"context_lines": 0})

    assert "+THREE" in result["diff"]
    assert "-THREE" not in result["diff"]
    # With zero context exactly one hunk of one changed line: one added, one removed.
    added = [
        line
        for line in result["diff"].splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line
        for line in result["diff"].splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    assert added == ["+THREE"]
    assert removed == ["-three"]


async def test_log_max_count(repo: Path) -> None:
    result = await GitLogTool().execute(git_context(repo), {"max_count": 1})

    assert len(result["logs"]) == 1
    assert result["logs"][0]["subject"] == "second commit"


async def test_log_path_filter(repo: Path) -> None:
    result = await GitLogTool().execute(git_context(repo), {"path": "a.txt"})

    subjects = [entry["subject"] for entry in result["logs"]]
    assert subjects == ["first commit"]
    assert all(entry["date"] for entry in result["logs"])
    assert all(entry["author"] == "Test User" for entry in result["logs"])


async def test_log_default_max_count(repo: Path) -> None:
    result = await GitLogTool().execute(git_context(repo), {})

    assert len(result["logs"]) == 2
    assert result["logs"][0]["hash"]
    assert re.fullmatch(r"[0-9a-f]{40}", result["logs"][0]["hash"])


async def test_show_commit(repo: Path) -> None:
    result = await GitShowTool().execute(git_context(repo), {"revision": "HEAD"})

    assert "second commit" in result["output"]
    assert "b.txt" in result["output"]
    assert "beta" in result["output"]  # without stat_only the patch body is present


async def test_show_stat_only(repo: Path) -> None:
    result = await GitShowTool().execute(git_context(repo), {"revision": "HEAD", "stat_only": True})

    assert "b.txt" in result["output"]
    # Diff body is not present when only the stat is requested.
    assert "beta\n" not in result["output"]


async def test_show_with_path(repo: Path) -> None:
    result = await GitShowTool().execute(git_context(repo), {"revision": "HEAD", "path": "b.txt"})

    assert "beta\n" in result["output"]


async def test_show_invalid_revision_starts_with_dash() -> None:
    with pytest.raises(ToolError):
        await GitShowTool().execute(git_context(Path(".")), {"revision": "-p"})


async def test_show_revision_with_option_like_chars_is_rejected() -> None:
    with pytest.raises(ToolError):
        await GitShowTool().execute(git_context(Path(".")), {"revision": "--output=x"})


async def test_show_revision_with_space_is_rejected() -> None:
    with pytest.raises(ToolError):
        await GitShowTool().execute(git_context(Path(".")), {"revision": "HEAD --output=x"})


async def test_show_empty_revision_is_rejected() -> None:
    with pytest.raises(ToolError):
        await GitShowTool().execute(git_context(Path(".")), {"revision": ""})


async def test_show_revision_with_newline_is_rejected() -> None:
    with pytest.raises(ToolError):
        await GitShowTool().execute(git_context(Path(".")), {"revision": "HEAD\n--output=x"})


async def test_diff_path_with_dotdot_is_rejected(repo: Path) -> None:
    with pytest.raises(ToolError):
        await GitDiffTool().execute(git_context(repo), {"paths": ["../outside.txt"]})


async def test_diff_path_starting_with_dash_is_rejected(repo: Path) -> None:
    with pytest.raises(ToolError):
        await GitDiffTool().execute(git_context(repo), {"paths": ["--all"]})


async def test_diff_path_empty_or_whitespace_is_rejected(repo: Path) -> None:
    with pytest.raises(ToolError):
        await GitDiffTool().execute(git_context(repo), {"paths": ["   "]})
    with pytest.raises(ToolError):
        await GitDiffTool().execute(git_context(repo), {"paths": [""]})


async def test_log_path_with_dotdot_is_rejected(repo: Path) -> None:
    with pytest.raises(ToolError):
        await GitLogTool().execute(git_context(repo), {"path": "sub/../escape"})


async def test_show_path_with_dotdot_is_rejected(repo: Path) -> None:
    with pytest.raises(ToolError):
        await GitShowTool().execute(
            git_context(repo), {"revision": "HEAD", "path": "../outside.txt"}
        )


async def test_diff_paths_capped_at_fifty(repo: Path) -> None:
    paths = [f"file{i}.txt" for i in range(60)]
    write(repo, "file0.txt", "zero\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "many files", "--no-gpg-sign")
    for i in range(60):
        write(repo, f"file{i}.txt", f"data {i}\n")

    result = await GitDiffTool().execute(git_context(repo), {"paths": paths})

    assert "data 0" in result["diff"]
    assert "data 59" not in result["diff"]


async def test_nonexistent_path_returns_empty_diff(repo: Path) -> None:
    result = await GitDiffTool().execute(git_context(repo), {"paths": ["no_such_file.txt"]})

    assert result["diff"] == ""


async def test_git_error_raises_tool_error(repo: Path) -> None:
    # A directory outside any git work tree: status must fail.
    not_a_repo = repo.parent / "outside"
    not_a_repo.mkdir()
    with pytest.raises(ToolError, match="Git error"):
        await GitStatusTool().execute(git_context(not_a_repo), {})


async def test_output_over_limit_is_truncated(repo: Path) -> None:
    # Produce output bigger than the cap: a blob with a huge number of changed lines.
    run_git(repo, "rm", "-q", "-r", ".")
    big = "".join(f"line {i}\n" for i in range(5000))
    write(repo, "big.txt", big)
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "big file", "--no-gpg-sign")
    changed = "".join(f"line {i} changed\n" for i in range(5000))
    write(repo, "big.txt", changed)

    result = await GitDiffTool().execute(git_context(repo), {})

    assert result["truncated"] is True
    assert len(result["diff"].encode("utf-8")) <= MAX_COMMAND_OUTPUT
    assert "line 4999 changed" not in result["diff"]


async def test_run_git_when_no_sandbox_and_repo(repo: Path) -> None:
    """_run_git returns the standard result shape for a successful command."""
    result = await _run_git(git_context(repo), ["status", "--porcelain=v1", "--branch"])
    assert {"stdout", "stderr", "truncated"} <= set(result)


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_cancelled_git_kills_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation kills the running git process and re-raises CancelledError."""
    import stat

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    wrapper = fakebin / "git"
    wrapper.write_text("#!/bin/sh\necho $$ > pid.txt\nsleep 30\n", encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    pid_file = project / "pid.txt"

    task = asyncio.create_task(
        _run_git(git_context(project), ["status", "--porcelain=v1", "--branch"])
    )
    # Wait until the fake git writes its pid. Under load the process can take a few
    # seconds to start; the limit only bounds the wait, it does not weaken the check.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while loop.time() < deadline:
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    child_pid = int(pid_file.read_text().strip())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The child must be gone, not left running as an orphan.
    deadline = loop.time() + 10
    while loop.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.02)
    os.kill(child_pid, 9)
    raise AssertionError("git process survived cancellation")


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_git_timeout_raises_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git command that does not finish in time raises ToolError."""
    import stat

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    wrapper = fakebin / "git"
    wrapper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")

    with pytest.raises(ToolError, match="timed out"):
        await _run_git(git_context(project), ["status"], timeout=1)


def test_revision_pattern_matches_valid_values() -> None:
    assert REVISION_PATTERN.fullmatch("HEAD~3") is not None
    assert REVISION_PATTERN.fullmatch("v1.2.3") is not None
    assert REVISION_PATTERN.fullmatch("abc123") is not None
    assert REVISION_PATTERN.fullmatch("HEAD") is not None


def test_revision_pattern_rejects_forbidden_values() -> None:
    # The char class itself allows "-p"; the tool adds an explicit startswith("-") guard.
    assert REVISION_PATTERN.fullmatch("--output=x") is None
    assert REVISION_PATTERN.fullmatch("HEAD --x") is None
    assert REVISION_PATTERN.fullmatch("") is None
    assert REVISION_PATTERN.fullmatch("HEAD\n") is None
    assert REVISION_PATTERN.fullmatch("HEAD\n--output=x") is None


def test_validate_paths_rejects_dotdot_and_options() -> None:
    with pytest.raises(ToolError):
        _validate_paths(["a/../b"])
    with pytest.raises(ToolError):
        _validate_paths(["--all"])
    with pytest.raises(ToolError):
        _validate_paths([""])
    with pytest.raises(ToolError):
        _validate_paths(["  "])
    assert _validate_paths(["a/b", "c.txt"]) == ["a/b", "c.txt"]


async def test_concurrent_status_and_diff_are_consistent(repo: Path) -> None:
    """Two concurrent readers see a stable snapshot; repeated runs do not duplicate."""
    a = b"".join(f"x {i}\n".encode() for i in range(200))
    (repo / "many.txt").write_bytes(a)
    run_git(repo, "add", "many.txt")

    async def run(tool: Any, args: dict[str, Any]) -> dict[str, Any]:
        return await tool.execute(git_context(repo), args)

    results = await asyncio.gather(
        run(GitStatusTool(), {}),
        run(GitDiffTool(), {"staged": True}),
        run(GitStatusTool(), {}),
        run(GitDiffTool(), {"staged": True}),
    )

    statuses = [result for result in results if "files" in result]
    assert len(statuses) == 2
    by_path = {item["path"]: (item["index"], item["worktree"]) for item in statuses[0]["files"]}
    assert by_path["many.txt"] == ("A", " ")
    # Repeated operation does not duplicate entries.
    assert all(len(result["files"]) == len(statuses[0]["files"]) for result in statuses)

    diffs = [result["diff"] for result in results if "diff" in result]
    assert diffs[0] == diffs[1]
    assert "x 199" in diffs[0]
    # Exactly 200 added lines, one per source line, no duplication from a repeat run.
    assert diffs[0].count("\n+x ") == 200


@pytest.mark.parametrize("tool", [GitLogTool(), GitShowTool()])
async def test_empty_path_is_rejected(repo: Path, tool: GitLogTool | GitShowTool) -> None:
    arguments: dict[str, object] = {"path": "  "}
    if isinstance(tool, GitShowTool):
        arguments["revision"] = "HEAD"
    with pytest.raises(ToolError):
        await tool.execute(git_context(repo), arguments)


def test_directory_path_with_trailing_slash_is_valid() -> None:
    assert _validate_paths(["src/"]) == ["src/"]
    with pytest.raises(ToolError):
        _validate_paths(["src//x"])


SECRET = "TOKEN-must-not-leak"


@pytest.fixture
def secret_repo(repo: Path) -> Path:
    """Commit secret-bearing files next to normal ones, then change all of them."""
    files = {
        ".env": SECRET,
        ".env.local": SECRET,
        "Config/.ENV": SECRET,
        "keys/id_ed25519": SECRET,
        "certs/server.KEY": SECRET,
        "credentials/token.txt": SECRET,
        ".env.example": "EXAMPLE=1",
    }
    for name, value in files.items():
        write(repo, name, f"{value}\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "add config", "--no-gpg-sign")
    for name, value in files.items():
        write(repo, name, f"{value} changed\n")
    write(repo, "a.txt", "alpha changed\n")
    return repo


async def test_diff_without_paths_hides_secret_files(secret_repo: Path) -> None:
    result = await GitDiffTool().execute(git_context(secret_repo), {})
    assert SECRET not in result["diff"]
    assert "EXAMPLE=1 changed" in result["diff"]  # .env.example stays visible
    assert "alpha changed" in result["diff"]


async def test_show_without_path_hides_secret_files(secret_repo: Path) -> None:
    result = await GitShowTool().execute(git_context(secret_repo), {"revision": "HEAD"})
    assert SECRET not in result["output"]
    assert "EXAMPLE=1" in result["output"]


@pytest.mark.parametrize(
    "path", [".env", ".env.local", "Config/.ENV", "keys/id_ed25519", "certs/server.KEY"]
)
async def test_explicit_secret_path_is_refused(secret_repo: Path, path: str) -> None:
    context = git_context(secret_repo)
    with pytest.raises(ToolError, match="secret-bearing"):
        await GitDiffTool().execute(context, {"paths": [path]})
    with pytest.raises(ToolError, match="secret-bearing"):
        await GitShowTool().execute(context, {"revision": "HEAD", "path": path})
    with pytest.raises(ToolError, match="secret-bearing"):
        await GitLogTool().execute(context, {"path": path})


async def test_directory_path_hides_secret_files_inside(secret_repo: Path) -> None:
    result = await GitDiffTool().execute(git_context(secret_repo), {"paths": ["keys/"]})
    assert SECRET not in result["diff"]


async def test_revision_with_path_suffix_is_refused(secret_repo: Path) -> None:
    with pytest.raises(ToolError, match="Invalid revision"):
        await GitShowTool().execute(git_context(secret_repo), {"revision": "HEAD:.env"})


async def test_status_marks_secret_files(secret_repo: Path) -> None:
    result = await GitStatusTool().execute(git_context(secret_repo), {})
    marks = {item["path"]: item["sensitive"] for item in result["files"]}
    assert marks[".env"] is True
    assert marks["keys/id_ed25519"] is True
    assert marks[".env.example"] is False
    assert marks["a.txt"] is False


def test_filter_keeps_normal_sections_and_trailing_newline() -> None:
    output = "diff --git a/.env b/.env\n+SECRET\ndiff --git a/a.txt b/a.txt\n+ok\n"
    assert _filter_diff_output(output) == (
        "[skipped: sensitive file]\ndiff --git a/a.txt b/a.txt\n+ok\n"
    )


async def test_show_merge_commit_hides_secret_combined_diff(repo: Path) -> None:
    # A conflict resolution in a merge commit is shown as a combined "diff --cc" section.
    write(repo, ".env.local", "SECRETV=base\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "base", "--no-gpg-sign")
    run_git(repo, "checkout", "-q", "-b", "side")
    write(repo, ".env.local", "SECRETV=side\n")
    run_git(repo, "commit", "-am", "side", "--no-gpg-sign")
    run_git(repo, "checkout", "-q", "main")
    write(repo, ".env.local", "SECRETV=main\n")
    run_git(repo, "commit", "-am", "main", "--no-gpg-sign")
    subprocess.run(["git", "merge", "-q", "side"], cwd=repo, capture_output=True)
    write(repo, ".env.local", f"SECRETV={SECRET}\n")
    run_git(repo, "add", ".env.local")
    run_git(repo, "commit", "-m", "merge", "--no-gpg-sign")

    result = await GitShowTool().execute(git_context(repo), {"revision": "HEAD"})

    assert SECRET not in result["output"]
    assert "[skipped: sensitive file]" in result["output"]


async def test_show_blob_revision_is_refused(secret_repo: Path) -> None:
    blob = run_git(secret_repo, "rev-parse", "HEAD:.env").stdout.strip()
    with pytest.raises(ToolError):
        await GitShowTool().execute(git_context(secret_repo), {"revision": blob})


@pytest.mark.parametrize(
    "header",
    [
        'diff --git "a/odd\\"name" "b/odd\\"name"',
        'diff --cc "odd\\tname"',
        "diff --git broken header",
    ],
)
def test_unparsed_header_is_hidden(header: str) -> None:
    assert _filter_diff_output(f"{header}\n+value\n") == "[skipped: sensitive file]\n"
