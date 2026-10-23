"""Split model output into the answer and a reasoning block marked by tags.

Some models write their reasoning into the answer text between markup tags instead
of a separate reasoning field. The tags come from the model configuration, so a
model without tags is never changed.
"""

from __future__ import annotations

from typing import Literal

Part = tuple[Literal["content", "reasoning"], str]


class ReasoningTagSplitter:
    """Split streamed text; a tag can arrive in several chunks."""

    def __init__(self, start: str, end: str) -> None:
        if not start or not end:
            raise ValueError("Reasoning tags must not be empty")
        self._start = start
        self._end = end
        self._inside = False
        self._held = ""

    def feed(self, text: str) -> list[Part]:
        """Return the parts that are known; keep a tail that can be the start of a tag."""
        buffer = self._held + text
        self._held = ""
        parts: list[Part] = []
        while buffer:
            tag = self._end if self._inside else self._start
            kind: Literal["content", "reasoning"] = "reasoning" if self._inside else "content"
            position = buffer.find(tag)
            if position >= 0:
                if position:
                    parts.append((kind, buffer[:position]))
                buffer = buffer[position + len(tag) :]
                self._inside = not self._inside
                continue
            keep = _partial_tag_length(buffer, tag)
            if len(buffer) > keep:
                parts.append((kind, buffer[: len(buffer) - keep]))
            self._held = buffer[len(buffer) - keep :]
            break
        return parts

    def close(self) -> list[Part]:
        """Return the held tail: the stream ended, so it is not a tag."""
        held, self._held = self._held, ""
        if not held:
            return []
        return [("reasoning" if self._inside else "content", held)]


def split_reasoning(text: str, start: str, end: str) -> tuple[str, str]:
    """Split a whole text; return (answer, reasoning)."""
    splitter = ReasoningTagSplitter(start, end)
    parts = splitter.feed(text) + splitter.close()
    content = "".join(part for kind, part in parts if kind == "content")
    reasoning = "".join(part for kind, part in parts if kind == "reasoning")
    return content, reasoning


def _partial_tag_length(text: str, tag: str) -> int:
    """Return the length of the longest end of ``text`` that starts ``tag``."""
    for length in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0
