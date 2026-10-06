from __future__ import annotations

import json

import pytest

from agent.llm import LLMError
from agent.sse import ChatStreamAccumulator, SSEDecoder, StreamEvent

# --- SSEDecoder -------------------------------------------------------------


def test_crlf_and_cr_line_endings() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b"data: one\r\ndata: two\r\n\r\n") == ["one\ntwo"]
    assert decoder.feed(b"data: three\r\r\n") == ["three"]
    assert decoder.feed(b"data: four\r") == []
    assert decoder.feed(b"data: five\n\n") == ["four\nfive"]


def test_comments_are_ignored() -> None:
    decoder = SSEDecoder()
    events = decoder.feed(b": hello\n: world\ndata: value\n\n: trailing\n")
    assert events == ["value"]


def test_multiline_data_joined_with_newline() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b"data: first\ndata: second\n\n") == ["first\nsecond"]


def test_split_across_chunks_line_and_utf8() -> None:
    decoder = SSEDecoder()
    payload = "data: привет мир\n\n".encode()
    events: list[str] = []
    for i in range(0, len(payload), 3):
        events.extend(decoder.feed(payload[i : i + 3]))
    assert events == ["привет мир"]


def test_close_flushes_last_event() -> None:
    decoder = SSEDecoder()
    decoder.feed(b"data: partial")
    assert decoder.close() == ["partial"]
    assert decoder.close() == []


def test_close_no_events() -> None:
    decoder = SSEDecoder()
    decoder.feed(b": comment\n")
    assert decoder.close() == []


# --- ChatStreamAccumulator --------------------------------------------------


def _chunk(payload: dict[str, object]) -> str:
    return json.dumps(payload)


def test_done_event() -> None:
    accumulator = ChatStreamAccumulator()
    assert accumulator.add("[DONE]") == [StreamEvent(kind="done")]


def test_content_and_reasoning_delta() -> None:
    accumulator = ChatStreamAccumulator()
    assert accumulator.add(_chunk({"choices": [{"delta": {"content": "Hel"}}]})) == [
        StreamEvent(kind="content", text="Hel")
    ]
    accumulator.add(_chunk({"choices": [{"delta": {"content": "lo"}}]}))
    reasoning = _chunk({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
    assert accumulator.add(reasoning) == [StreamEvent(kind="reasoning", text="thinking")]
    accumulator.add(_chunk({"choices": [{"delta": {"reasoning": " more"}}]}))
    result = accumulator.result()
    assert result.content == "Hello"
    assert result.reasoning == "thinking more"


def test_null_and_empty_delta_fields_skipped() -> None:
    accumulator = ChatStreamAccumulator()
    events = accumulator.add(
        _chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "reasoning_content": None,
                            "reasoning": "",
                            "tool_calls": None,
                        }
                    }
                ]
            }
        )
    )
    assert events == []
    assert accumulator.result().content is None
    assert accumulator.result().reasoning is None


def test_empty_choices_allowed() -> None:
    accumulator = ChatStreamAccumulator()
    events = accumulator.add(_chunk({"choices": []}))
    assert events == []
    assert accumulator.result().content is None


def test_usage_only_chunk() -> None:
    accumulator = ChatStreamAccumulator()
    usage = _chunk(
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    assert accumulator.add(usage) == [StreamEvent(kind="usage")]
    result = accumulator.result()
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_finish_reason_event_and_default() -> None:
    accumulator = ChatStreamAccumulator()
    assert accumulator.add(_chunk({"choices": [{"finish_reason": "tool_calls"}]})) == [
        StreamEvent(kind="finish")
    ]
    assert accumulator.result().finish_reason == "tool_calls"
    assert ChatStreamAccumulator().result().finish_reason == "stop"


def test_tool_calls_interleaved_and_concatenated() -> None:
    accumulator = ChatStreamAccumulator()
    accumulator.add(
        _chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call-b",
                                    "type": "function",
                                    "function": {"name": "tool_b", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    accumulator.add(
        _chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "tool_a", "arguments": '{"p": '},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    accumulator.add(
        _chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "function": {"arguments": "1"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    accumulator.add(
        _chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "1}"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    accumulator.add(_chunk({"choices": [{"finish_reason": "tool_calls"}]}))
    result = accumulator.result()
    assert result.finish_reason == "tool_calls"
    assert [call["id"] for call in result.tool_calls] == [None, "call-b"]
    assert [call["type"] for call in result.tool_calls] == ["function", "function"]
    assert [call["function"]["name"] for call in result.tool_calls] == [
        "tool_a",
        "tool_b",
    ]
    assert [call["function"]["arguments"] for call in result.tool_calls] == [
        '{"p": 1}',
        "1",
    ]


def test_tool_call_events_have_index() -> None:
    accumulator = ChatStreamAccumulator()
    events = accumulator.add(
        _chunk({"choices": [{"delta": {"tool_calls": [{"index": 2, "function": {"name": "x"}}]}}]})
    )
    assert events == [StreamEvent(kind="tool_call", index=2)]


def test_error_after_successful_chunks() -> None:
    accumulator = ChatStreamAccumulator()
    accumulator.add(_chunk({"choices": [{"delta": {"content": "partial"}}]}))
    with pytest.raises(LLMError, match="boom"):
        accumulator.add(_chunk({"error": {"message": "boom"}}))


def test_result_no_usage_defaults_zero() -> None:
    accumulator = ChatStreamAccumulator()
    accumulator.add(_chunk({"choices": [{"delta": {"content": "hi"}}]}))
    result = accumulator.result()
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_full_stream_result() -> None:
    accumulator = ChatStreamAccumulator()
    accumulator.add(_chunk({"choices": [{"delta": {"content": "Answer"}}]}))
    accumulator.add(
        _chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    accumulator.add(
        _chunk(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}}]}
        )
    )
    accumulator.add(
        _chunk(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7},
            }
        )
    )
    result = accumulator.result()
    assert result.content == "Answer"
    assert result.tool_calls == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
    ]
    assert result.finish_reason == "tool_calls"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 7


def test_crlf_split_between_chunks_does_not_end_the_event() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b"data: one\r") == []
    assert decoder.feed(b"\ndata: two\r\n\r\n") == ["one\ntwo"]


def test_invalid_utf8_is_replaced() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b"data: \xff\n\n") == ["\ufffd"]
