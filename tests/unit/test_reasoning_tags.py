from __future__ import annotations

import pytest

from agent.reasoning_tags import ReasoningTagSplitter, split_reasoning

# Example delimiters for a reasoning block embedded in answer text.
START = "<|channel>thought"
END = "<channel|>"


def _run(chunks: list[str]) -> tuple[str, str]:
    splitter = ReasoningTagSplitter(START, END)
    parts = []
    for chunk in chunks:
        parts.extend(splitter.feed(chunk))
    parts.extend(splitter.close())
    content = "".join(text for kind, text in parts if kind == "content")
    reasoning = "".join(text for kind, text in parts if kind == "reasoning")
    return content, reasoning


def test_block_in_one_chunk_goes_to_reasoning() -> None:
    assert _run(["<|channel>thought plan<channel|>Answer"]) == ("Answer", " plan")


def test_empty_block_after_a_tool_result_is_removed_from_the_answer() -> None:
    assert _run(["<|channel>thought <channel|>Ответ"]) == ("Ответ", " ")


def test_tags_split_across_chunks() -> None:
    chunks = ["<|chan", "nel>thou", "ght pla", "n<chan", "nel|>An", "swer"]
    assert _run(chunks) == ("Answer", " plan")


def test_text_without_tags_is_unchanged() -> None:
    assert _run(["Hello ", "world"]) == ("Hello world", "")


def test_a_tag_prefix_that_never_becomes_a_tag_stays_in_the_answer() -> None:
    # The stream ends while the text still looks like the start of a tag.
    assert _run(["a <|ch"]) == ("a <|ch", "")
    # The next character shows that it is not a tag.
    assert _run(["a <", "x"]) == ("a <x", "")


def test_unclosed_block_stays_reasoning() -> None:
    assert _run(["<|channel>thought still thinking"]) == ("", " still thinking")


def test_several_blocks_are_all_removed() -> None:
    text = "A<|channel>thought x<channel|>B<|channel>thought y<channel|>C"
    assert _run([text]) == ("ABC", " x y")


def test_split_reasoning_for_a_whole_text() -> None:
    assert split_reasoning("<|channel>thought p<channel|>Ok", START, END) == ("Ok", " p")
    assert split_reasoning("plain", START, END) == ("plain", "")


def test_empty_tags_are_refused() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ReasoningTagSplitter("", END)
    with pytest.raises(ValueError, match="must not be empty"):
        ReasoningTagSplitter(START, "")
