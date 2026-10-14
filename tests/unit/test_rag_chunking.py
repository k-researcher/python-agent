"""Tests for the RAG document chunking helpers."""

from __future__ import annotations

import random

import pytest

from agent.rag.chunking import (
    Chunk,
    chunk_document,
    chunk_markdown,
    chunk_python,
    chunk_text,
)


def _verify_invariants(
    chunks: list[Chunk],
    source_text: str,
    lines: list[str],
    max_chars: int,
    *,
    overlap: bool = False,
) -> None:
    """Check that all chunking invariants hold."""
    for chunk in chunks:
        assert len(chunk.text) <= max_chars, (
            f"chunk length {len(chunk.text)} > max_chars {max_chars}"
        )
        expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        if chunk.start_line == chunk.end_line and len(expected) > max_chars:
            # Character-split chunks are substrings of a long line.
            assert chunk.text in expected
        else:
            assert chunk.text == expected, (
                f"text mismatch for lines {chunk.start_line}-{chunk.end_line}"
            )

    if not overlap and chunks:
        # Non-empty lines must be covered without gaps.
        covered: set[int] = set()
        for chunk in chunks:
            covered.update(range(chunk.start_line, chunk.end_line + 1))
        non_empty = {i + 1 for i, line in enumerate(lines) if line.strip() != ""}
        assert non_empty <= covered, f"lines not covered: {non_empty - covered}"


def test_empty_text() -> None:
    assert chunk_text("", "doc.txt") == []
    assert chunk_text("   \n  ", "doc.txt") == []
    assert chunk_markdown("", "doc.md") == []
    assert chunk_python("", "mod.py") == []
    assert chunk_document("", "doc.txt") == []


def test_max_chars_validation() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("text", "doc.txt", max_chars=0)
    with pytest.raises(ValueError, match="max_chars"):
        chunk_markdown("text", "doc.md", max_chars=-1)
    with pytest.raises(ValueError, match="max_chars"):
        chunk_python("text", "mod.py", max_chars=0)


def test_single_line_text() -> None:
    chunks = chunk_text("hello world", "doc.txt", max_chars=100)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == "hello world"
    assert c.source == "doc.txt"
    assert c.start_line == 1
    assert c.end_line == 1
    assert c.heading_path == ()
    assert c.kind == "text"


def test_chunk_text_multi_line() -> None:
    text = "line one\nline two\nline three"
    chunks = chunk_text(text, "doc.txt", max_chars=100)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == text
    assert c.start_line == 1
    assert c.end_line == 3


def test_chunk_text_paragraph_split() -> None:
    lines = [
        "AAA",
        "BBB",
        "",
        "CCC",
        "DDD",
    ]
    text = "\n".join(lines)
    chunks = chunk_text(text, "doc.txt", max_chars=15)
    # Each paragraph is short enough alone but not together.
    assert len(chunks) == 2
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2
    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 5
    _verify_invariants(chunks, text, lines, 15)


def test_chunk_text_long_line_char_split() -> None:
    line = "x" * 300
    chunks = chunk_text(line, "doc.txt", max_chars=100)
    assert len(chunks) == 3
    assert all(len(c.text) == 100 for c in chunks)
    assert all(c.start_line == 1 and c.end_line == 1 for c in chunks)
    # Reconstruct the full line from character pieces.
    assert "".join(c.text for c in chunks) == line
    _verify_invariants(chunks, line, [line], 100)


def test_chunk_text_crlf() -> None:
    text = "one\r\ntwo\r\nthree"
    chunks = chunk_text(text, "doc.txt", max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == "one\ntwo\nthree"


def test_chunk_text_no_trailing_newline() -> None:
    text = "one\ntwo\n"  # trailing newline
    chunks = chunk_text(text, "doc.txt", max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == "one\ntwo"
    assert chunks[0].end_line == 2


def test_chunk_text_coverage_without_overlap() -> None:
    text = "Short paragraph.\n\nAnother short paragraph.\nWith a second line.\n\nThird one."
    lines = text.split("\n")
    # 6 lines: paragraph1, blank, p2l1, p2l2, blank, p3.
    assert len(lines) == 6, f"expected 6 lines, got {len(lines)}"
    chunks = chunk_text(text, "doc.txt", max_chars=25, overlap_lines=0)
    _verify_invariants(chunks, text, lines, 25)

    # Text should be covered without gap or overlap.
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 6


def test_markdown_heading_path() -> None:
    text = "# Top Header\nSome content under top.\n\n## Sub Header\nContent under sub.\n"
    chunks = chunk_markdown(text, "doc.md", max_chars=100)
    assert len(chunks) >= 2
    top = [c for c in chunks if c.heading_path == ("Top Header",)]
    sub = [c for c in chunks if c.heading_path == ("Top Header", "Sub Header")]
    assert top, "missing top-level section"
    assert sub, "missing sub-level section"
    assert top[0].kind == "markdown"
    assert sub[0].kind == "markdown"


def test_markdown_code_block_preserved() -> None:
    code = "```python\ndef foo():\n    return 1\n```"
    text = "# Title\nSome text\n\n" + code + "\n\nMore text"
    chunks = chunk_markdown(text, "doc.md", max_chars=200)
    code_chunks = [c for c in chunks if c.kind == "code"]
    assert code_chunks, "code block not preserved as a chunk"
    assert code in code_chunks[0].text
    _verify_invariants(chunks, text, text.split("\n"), 200)


def test_markdown_long_code_block_split() -> None:
    code_lines = ["```python"]
    code_lines.extend(f"    line_{i} = {i}" for i in range(50))
    code_lines.append("```")
    code = "\n".join(code_lines)
    text = code
    chunks = chunk_markdown(text, "doc.md", max_chars=100)
    code_chunks = [c for c in chunks if c.kind == "code"]
    assert code_chunks, "expected code chunks"
    assert all(c.kind == "code" for c in chunks)
    _verify_invariants(chunks, text, text.split("\n"), 100)


def test_markdown_section_split() -> None:
    paragraph = "This is a line that is long enough.\n"
    text = "# Header\n" + paragraph * 100
    chunks = chunk_markdown(text, "doc.md", max_chars=100)
    assert len(chunks) > 1
    assert all(c.heading_path == ("Header",) for c in chunks)
    _verify_invariants(chunks, text, text.split("\n"), 100)


def test_python_top_level_defs() -> None:
    code = (
        "import os\n"
        "\n"
        "def first():\n"
        '    """Docstring."""\n'
        "    return 1\n"
        "\n"
        "class MyClass:\n"
        "    pass\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )
    chunks = chunk_python(code, "mod.py", max_chars=200)
    assert len(chunks) >= 4  # import code, first, class, second
    kinds = [c.kind for c in chunks]
    assert all(k == "code" for k in kinds)
    _verify_invariants(chunks, code, code.split("\n"), 200)


def test_python_decorators() -> None:
    code = "@decorator1\n@decorator2\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
    chunks = chunk_python(code, "mod.py", max_chars=200)
    assert len(chunks) >= 2
    # First chunk should include decorators.
    foo_chunk = chunks[0]
    assert "@decorator1" in foo_chunk.text
    assert "def foo" in foo_chunk.text
    assert foo_chunk.start_line == 1  # decorators start at line 1
    assert foo_chunk.end_line == 4
    _verify_invariants(chunks, code, code.split("\n"), 200)


def test_python_syntax_error_fallback() -> None:
    code = "def broken(:\n    pass\n"
    chunks = chunk_python(code, "mod.py", max_chars=200)
    # Falls back to chunk_text, producing "text" kind chunks.
    assert len(chunks) >= 1
    assert all(c.kind == "text" for c in chunks)
    _verify_invariants(chunks, code, code.split("\n"), 200)


def test_python_docstring_included() -> None:
    code = 'def foo():\n    """A docstring that explains the function."""\n    return 42\n'
    chunks = chunk_python(code, "mod.py", max_chars=200)
    assert len(chunks) == 1
    assert "docstring" in chunks[0].text
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


def test_chunk_document_routing() -> None:
    md = "# Title\nbody"
    py = "def foo():\n    pass"
    txt = "plain text"
    assert chunk_document(md, "file.md") == chunk_markdown(md, "file.md")
    assert chunk_document(md, "file.MARKDOWN") == chunk_markdown(md, "file.MARKDOWN")
    assert chunk_document(py, "file.py") == chunk_python(py, "file.py")
    assert chunk_document(txt, "file.txt") == chunk_text(txt, "file.txt")
    assert chunk_document(txt, "file.unknown") == chunk_text(txt, "file.unknown")


def test_overlap_lines() -> None:
    # Lines "ab" fit 2 per chunk with max_chars=7.
    text = "ab\nab\nab\nab\nab\n"
    no_overlap = chunk_text(text, "doc.txt", max_chars=7, overlap_lines=0)
    with_overlap = chunk_text(text, "doc.txt", max_chars=7, overlap_lines=1)

    assert len(with_overlap) >= len(no_overlap)
    # No-overlap chunks are sequential without gaps.
    for prev, nxt in zip(no_overlap, no_overlap[1:], strict=False):
        assert nxt.start_line == prev.end_line + 1
    # Overlap chunks repeat the boundary line.
    if len(with_overlap) >= 2:
        assert with_overlap[1].start_line == with_overlap[0].end_line

    # With overlap_lines=0, every non-empty line is covered exactly once.
    covered = set()
    for c in no_overlap:
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered == {1, 2, 3, 4, 5}


def test_chunk_is_frozen() -> None:
    chunk = chunk_text("text", "doc.txt")[0]
    with pytest.raises(AttributeError):
        chunk.text = "changed"  # type: ignore[misc]


def test_random_inputs() -> None:
    rng = random.Random(42)
    for _trial in range(30):
        num_lines = rng.randint(1, 15)
        lines = []
        for _ in range(num_lines):
            if rng.random() < 0.25:
                lines.append("")
            else:
                length = rng.randint(1, 200)
                lines.append("".join(rng.choice("abc ") for _ in range(length)))
        text = "\n".join(lines)
        max_chars = rng.randint(5, 120)

        for chunker in (chunk_text, chunk_markdown):
            chunks = chunker(text, "doc.md", max_chars=max_chars, overlap_lines=0)
            assert all(len(c.text) <= max_chars for c in chunks)
            for c in chunks:
                expected = "\n".join(lines[c.start_line - 1 : c.end_line])
                if c.start_line == c.end_line and len(expected) > max_chars:
                    assert c.text in expected
                else:
                    assert c.text == expected
            # Coverage of non-empty text.
            non_empty = {i + 1 for i, ln in enumerate(lines) if ln.strip()}
            covered = set()
            for c in chunks:
                covered.update(range(c.start_line, c.end_line + 1))
            assert non_empty <= covered

        chunks = chunk_python(text, "mod.py", max_chars=max_chars)
        assert all(len(c.text) <= max_chars for c in chunks)


def test_heading_inside_code_fence_does_not_split_the_block() -> None:
    text = (
        "# Title\n\n```python\n# comment inside code\ndef foo():\n    return 1\n```\n\nAfter text\n"
    )
    chunks = chunk_markdown(text, "doc.md")
    lines = text.splitlines()
    fence_start = lines.index("```python") + 1
    fence_end = len(lines) - 2  # the closing fence line
    code = [c for c in chunks if c.start_line <= fence_start and c.end_line >= fence_end]
    assert len(code) == 1  # the whole block, both fences, lives in one chunk
    assert "# comment inside code" in code[0].text
    assert all(c.heading_path == ("Title",) for c in chunks if c.start_line > 1)
