"""Chunk text documents for retrieval-augmented generation.

This module provides chunking helpers. Each helper splits text into
small pieces that stay within a size limit.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_PATTERN = re.compile(r"^```")
BLANK_LINE = re.compile(r"^\s*$")


@dataclass(frozen=True)
class Chunk:
    """A piece of a source document."""

    text: str
    source: str
    start_line: int
    end_line: int
    heading_path: tuple[str, ...]
    kind: str


def _normalized(text: str) -> str:
    """Return text with CRLF converted to LF and no trailing newline."""
    if not text:
        return ""
    return text.replace("\r\n", "\n").rstrip("\n")


def _lines(text: str) -> list[str]:
    """Split text into lines without the line separator."""
    if not text:
        return []
    return text.split("\n")


def _join(start: int, end: int, lines: list[str]) -> str:
    """Join lines start..end inclusive, 1-based, without trailing newline."""
    return "\n".join(lines[start - 1 : end])


def _line_len(line: str) -> int:
    """Return line length plus one for the separator."""
    return len(line) + 1


def _split_by_char(text: str, max_chars: int) -> list[str]:
    """Split text into fixed-size pieces by character count."""
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _emit(
    lines: list[str],
    source: str,
    start: int,
    end: int,
    heading: tuple[str, ...],
    kind: str,
) -> Chunk:
    """Build a Chunk from a 1-based inclusive line range."""
    return Chunk(
        text=_join(start, end, lines),
        source=source,
        start_line=start,
        end_line=end,
        heading_path=heading,
        kind=kind,
    )


def _line_chunks(
    lines: list[str],
    source: str,
    start: int,
    end: int,
    heading: tuple[str, ...],
    max_chars: int,
    overlap_lines: int,
    kind: str,
) -> list[Chunk]:
    """Split lines into sequential chunks within the character limit."""
    if start > end:
        return []
    chunks: list[Chunk] = []
    cursor = start
    while cursor <= end:
        chunk_start = cursor
        total = _line_len(lines[cursor - 1])
        chunk_end = cursor
        while chunk_end < end:
            next_len = _line_len(lines[chunk_end])
            if total + next_len > max_chars:
                break
            total += next_len
            chunk_end += 1

        # A single over-long line gets split by characters.
        if chunk_end == chunk_start:
            line_text = lines[chunk_start - 1]
            if len(line_text) > max_chars:
                for piece in _split_by_char(line_text, max_chars):
                    chunks.append(
                        Chunk(
                            text=piece,
                            source=source,
                            start_line=chunk_start,
                            end_line=chunk_end,
                            heading_path=heading,
                            kind=kind,
                        )
                    )
                cursor = chunk_start + 1
                continue

        chunks.append(_emit(lines, source, chunk_start, chunk_end, heading, kind))
        if chunk_end >= end:
            break
        cursor = max(chunk_end - overlap_lines + 1, chunk_start + 1)
    return chunks


def _paragraph_ranges(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Return inclusive 1-based ranges for each paragraph."""
    if start > end:
        return []
    ranges: list[tuple[int, int]] = []
    group_start: int | None = None
    for line_no in range(start, end + 1):
        if not BLANK_LINE.match(lines[line_no - 1]):
            if group_start is None:
                group_start = line_no
        else:
            if group_start is not None:
                ranges.append((group_start, line_no - 1))
                group_start = None
    if group_start is not None:
        ranges.append((group_start, end))
    return ranges


def _paragraph_chunks(
    lines: list[str],
    source: str,
    start: int,
    end: int,
    heading: tuple[str, ...],
    max_chars: int,
    overlap_lines: int,
    kind: str,
) -> list[Chunk]:
    """Split text by paragraphs first, then by lines when needed."""
    if start > end:
        return []
    ranges = _paragraph_ranges(lines, start, end)
    if not ranges:
        return []

    chunks: list[Chunk] = []
    current_start = ranges[0][0]
    current_end = ranges[0][1]
    current_len = len(_join(current_start, current_end, lines)) + 1

    for group_start, group_end in ranges[1:]:
        group_len = len(_join(group_start, group_end, lines)) + 1
        if current_len + group_len > max_chars:
            chunks.extend(
                _line_chunks(
                    lines,
                    source,
                    current_start,
                    current_end,
                    heading,
                    max_chars,
                    overlap_lines,
                    kind,
                )
            )
            current_start = group_start
            current_end = group_end
            current_len = group_len
        else:
            current_end = group_end
            current_len += group_len

    chunks.extend(
        _line_chunks(
            lines,
            source,
            current_start,
            current_end,
            heading,
            max_chars,
            overlap_lines,
            kind,
        )
    )
    return chunks


def chunk_text(
    text: str,
    source: str,
    max_chars: int = 2000,
    overlap_lines: int = 2,
) -> list[Chunk]:
    """Split plain text into chunks by paragraphs, then by lines."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    normalized = _normalized(text)
    if not normalized.strip():
        return []
    lines = _lines(normalized)
    return _paragraph_chunks(lines, source, 1, len(lines), (), max_chars, overlap_lines, "text")


def _split_by_heading(lines: list[str]) -> list[tuple[int, int, tuple[str, ...]]]:
    """Split markdown lines into sections by heading boundaries.

    Returns flat entries of (start, end, heading_path) where start and
    end are 1-based inclusive line numbers.
    """
    sections: list[tuple[int, int, tuple[str, ...]]] = []
    current_path: tuple[str, ...] = ()

    # Find all heading line numbers and their heading paths.
    heading_entries: list[tuple[int, int, tuple[str, ...]]] = []
    in_fence = False
    for line_no, line in enumerate(lines, start=1):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            continue
        # A "# ..." line inside a fenced code block is code, not a heading.
        match = None if in_fence else HEADING_PATTERN.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            current_path = (title,) if level == 1 else current_path[: level - 1] + (title,)
            heading_entries.append((line_no, level, current_path))

    if not heading_entries:
        return [(1, len(lines), ())] if lines else []

    # First section: content before the first heading (may be empty).
    first_heading_line = heading_entries[0][0]
    if first_heading_line > 1:
        sections.append((1, first_heading_line - 1, ()))

    for idx, (line_no, _level, path) in enumerate(heading_entries):
        next_entry = heading_entries[idx + 1][0] if idx + 1 < len(heading_entries) else None
        next_line = next_entry if next_entry is not None else len(lines) + 1
        end = next_line - 1
        # The heading line itself starts a section, so we include it.
        sections.append((line_no, end, path))

    return sections


def _code_blocks(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Return inclusive line ranges for fenced code blocks.

    A block starts at the opening fence and ends at the closing fence.
    An unclosed block ends at the section end.
    """
    blocks: list[tuple[int, int]] = []
    in_block = False
    block_start = start
    for line_no in range(start, end + 1):
        if FENCE_PATTERN.match(lines[line_no - 1]):
            if not in_block:
                in_block = True
                block_start = line_no
            else:
                in_block = False
                blocks.append((block_start, line_no))
    if in_block:
        blocks.append((block_start, end))
    return blocks


def chunk_markdown(
    text: str,
    source: str,
    max_chars: int = 2000,
    overlap_lines: int = 2,
) -> list[Chunk]:
    """Split markdown into chunks by headings, keeping code blocks intact."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    normalized = _normalized(text)
    if not normalized.strip():
        return []
    lines = _lines(normalized)
    if not lines:
        return []

    sections = _split_by_heading(lines)
    chunks: list[Chunk] = []

    for sec_start, sec_end, sec_path in sections:
        if sec_start > sec_end:
            continue
        # Skip sections that are only whitespace.
        section_text = _join(sec_start, sec_end, lines)
        if not section_text.strip():
            continue

        # Sections with code blocks need special handling.
        blocks = _code_blocks(lines, sec_start, sec_end)
        if blocks:
            cursor = sec_start
            for block_start, block_end in blocks:
                if cursor < block_start:
                    chunks.extend(
                        _paragraph_chunks(
                            lines,
                            source,
                            cursor,
                            block_start - 1,
                            sec_path,
                            max_chars,
                            overlap_lines,
                            "markdown",
                        )
                    )
                block_text = _join(block_start, block_end, lines)
                if len(block_text) + 1 <= max_chars:
                    chunks.append(_emit(lines, source, block_start, block_end, sec_path, "code"))
                else:
                    chunks.extend(
                        _line_chunks(
                            lines,
                            source,
                            block_start,
                            block_end,
                            sec_path,
                            max_chars,
                            overlap_lines,
                            "code",
                        )
                    )
                cursor = block_end + 1
            if cursor <= sec_end:
                chunks.extend(
                    _paragraph_chunks(
                        lines,
                        source,
                        cursor,
                        sec_end,
                        sec_path,
                        max_chars,
                        overlap_lines,
                        "markdown",
                    )
                )
        else:
            chunks.extend(
                _paragraph_chunks(
                    lines,
                    source,
                    sec_start,
                    sec_end,
                    sec_path,
                    max_chars,
                    overlap_lines,
                    "markdown",
                )
            )

    return chunks


def _top_level_nodes(tree: ast.Module) -> list[tuple[int, int]]:
    """Return (start_line, end_line) inclusive for each top-level def/class.

    The start line accounts for decorators.
    """
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef):
            start = node.lineno
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list))
            end = node.end_lineno or node.lineno
            ranges.append((start, end))
    return ranges


def chunk_python(
    text: str,
    source: str,
    max_chars: int = 2000,
) -> list[Chunk]:
    """Split Python code by top-level def/class using the AST."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    normalized = _normalized(text)
    if not normalized.strip():
        return []
    lines = _lines(normalized)
    if not lines:
        return []

    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return chunk_text(text, source, max_chars)

    ranges = _top_level_nodes(tree)
    if not ranges:
        return chunk_text(text, source, max_chars)

    chunks: list[Chunk] = []
    cursor = 1
    for start, end in ranges:
        # Code between the previous definition and this one.
        if cursor < start:
            chunks.extend(
                _paragraph_chunks(
                    lines,
                    source,
                    cursor,
                    start - 1,
                    (),
                    max_chars,
                    2,
                    "code",
                )
            )
        chunks.extend(
            _line_chunks(
                lines,
                source,
                start,
                end,
                (),
                max_chars,
                2,
                "code",
            )
        )
        cursor = end + 1

    # Trailing module-level code.
    if cursor <= len(lines):
        chunks.extend(
            _paragraph_chunks(
                lines,
                source,
                cursor,
                len(lines),
                (),
                max_chars,
                2,
                "code",
            )
        )

    return chunks


def chunk_document(text: str, source: str) -> list[Chunk]:
    """Route text to the correct chunker based on the source extension."""
    lowered = source.lower()
    if lowered.endswith((".md", ".markdown")):
        return chunk_markdown(text, source)
    if lowered.endswith(".py"):
        return chunk_python(text, source)
    return chunk_text(text, source)
