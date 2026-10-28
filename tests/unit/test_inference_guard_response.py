"""Framing, finish priority and validation of original response dialects."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from agent.llm import LLMClient, LLMError
from agent.sse import ChatStreamAccumulator, SSEDecoder
from tests.inference_guard_helpers import guard_tasks as guard_tasks
from tests.unit.test_llm_send import _client, _model, _request, _sse

pytestmark = pytest.mark.usefixtures("guard_tasks")


def add(acc, delta=None, **choice):
    acc.add(json.dumps({"choices": [{"delta": delta or {}, **choice}]}))


@pytest.mark.parametrize(
    "finish",
    ["stop", "tool_calls", "length", "content_filter", "unknown", 17, "conflict", None, "absent"],
)
@pytest.mark.parametrize("done", [False, True])
async def test_finish_boundary_matrix(finish, done):
    async with asyncio.timeout(5):
        acc = ChatStreamAccumulator()
        delta = {"content": "answer"}
        if finish == "tool_calls":
            delta["tool_calls"] = [
                {"index": 0, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]
        if finish == "conflict":
            add(acc, delta, finish_reason="stop")
            with pytest.raises(LLMError) as caught:
                add(acc, {}, finish_reason="length")
            assert caught.value.kind.value == "llm_invalid_response"
            return
        if finish in ("unknown", 17, "content_filter"):
            with pytest.raises(LLMError) as caught:
                add(acc, delta, finish_reason=finish)
            assert caught.value.kind.value == "llm_invalid_response"
            return
        add(acc, delta, **({} if finish == "absent" else {"finish_reason": finish}))
        if done:
            acc.add("[DONE]")
        invalid = finish in ("unknown", 17)
        incomplete = finish in (None, "absent") and not done
        if invalid or incomplete:
            with pytest.raises(LLMError) as caught:
                acc.finalize()
            assert caught.value.kind.value == (
                "llm_invalid_response" if invalid else "llm_stream_incomplete"
            )
        else:
            assert acc.finalize().finish_reason == (
                "stop" if finish in (None, "absent") else finish
            )


@pytest.mark.parametrize("delta", [{}, {"content": " \n"}, {"reasoning_content": "thinking"}])
async def test_empty_after_normal_stop(delta):
    async with asyncio.timeout(5):
        acc = ChatStreamAccumulator()
        add(acc, delta, finish_reason="stop")
        with pytest.raises(LLMError) as caught:
            acc.finalize()
        assert caught.value.kind.value == "llm_empty_response"


@pytest.mark.parametrize(
    "delta",
    [
        {},
        {"reasoning_content": "thinking"},
        {"tool_calls": [{"index": 0, "function": {"name": "read", "arguments": '{"path":'}}]},
    ],
)
async def test_length_precedes_empty_and_unfinished_arguments(delta):
    async with asyncio.timeout(5):
        acc = ChatStreamAccumulator()
        add(acc, delta, finish_reason="length")
        result = acc.finalize()
        assert result.finish_reason == "length"
        assert result.tool_calls == []


@pytest.mark.parametrize("bad", [None, [], 3, "", "{", "[]", "1"])
@pytest.mark.parametrize("stream", [False, True])
async def test_functions_arguments_invalid_in_original_dialect(bad, stream):
    async with asyncio.timeout(5):
        model = replace(_model(max_retries=3), dialect="functions")
        call = {"name": "read", "arguments": bad}
        payload = {
            "choices": [
                {
                    "delta" if stream else "message": {"function_call": call},
                    "finish_reason": "function_call",
                }
            ]
        }
        calls = []

        def handler(request):
            calls.append(request)
            return (
                httpx.Response(200, content=_sse(payload) + b"data: [DONE]\n\n")
                if stream
                else httpx.Response(200, json=payload)
            )

        client = _client(model, handler)

        async def delta(kind, text):
            pass

        with pytest.raises(LLMError) as caught:
            await client.send(_request(model, stream=stream), on_delta=delta if stream else None)
        assert caught.value.kind.value == "llm_invalid_response"
        assert len(calls) == 1
        await client.aclose()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("call", [{"name": "read"}, None, [], "bad"])
async def test_functions_missing_arguments_or_wrong_call(stream, call):
    async with asyncio.timeout(5):
        model = replace(_model(), dialect="functions")
        payload = {
            "choices": [
                {
                    "delta" if stream else "message": {"function_call": call},
                    "finish_reason": "function_call",
                }
            ]
        }
        client = _client(
            model,
            lambda _: (
                httpx.Response(200, content=_sse(payload) + b"data: [DONE]\n\n")
                if stream
                else httpx.Response(200, json=payload)
            ),
        )

        async def delta(kind, text):
            pass

        with pytest.raises(LLMError) as caught:
            await client.send(_request(model, stream=stream), on_delta=delta if stream else None)
        assert caught.value.kind.value == "llm_invalid_response"
        await client.aclose()


@pytest.mark.parametrize("value", ["{}", {"path": "a"}])
@pytest.mark.parametrize("stream", [False, True])
async def test_functions_valid_arguments(value, stream):
    async with asyncio.timeout(5):
        model = replace(_model(), dialect="functions")
        payload = {
            "choices": [
                {
                    "delta" if stream else "message": {
                        "function_call": {"name": "read", "arguments": value}
                    },
                    "finish_reason": "function_call",
                }
            ]
        }
        client = _client(
            model,
            lambda _: (
                httpx.Response(200, content=_sse(payload) + b"data: [DONE]\n\n")
                if stream
                else httpx.Response(200, json=payload)
            ),
        )

        async def delta(kind, text):
            pass

        result = await client.send(
            _request(model, stream=stream), on_delta=delta if stream else None
        )
        assert json.loads(result.tool_calls[0]["function"]["arguments"]) == (
            json.loads(value) if isinstance(value, str) else value
        )
        await client.aclose()


async def test_done_stops_before_tail_and_without_eof():
    async with asyncio.timeout(5):

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield (
                    _sse({"choices": [{"delta": {"content": "answer"}}]})
                    + b"data: [DONE]\n\n"
                    + b"x" * 300000
                )
                pytest.fail("Reading after DONE")

        client = _client(_model(), lambda _: httpx.Response(200, stream=Stream()))

        async def delta(kind, text):
            pass

        assert (
            await client.send(_request(client.model, stream=True), on_delta=delta)
        ).content == "answer"
        await client.aclose()


@pytest.mark.parametrize("size,valid", [(262144, True), (262145, False)])
async def test_event_byte_limit(size, valid):
    async with asyncio.timeout(5):
        decoder = SSEDecoder()
        data = b":" + b"x" * (size - 2) + b"\n" + b"\n"
        if valid:
            assert list(decoder.feed(data)) == []
        else:
            with pytest.raises(LLMError) as caught:
                list(decoder.feed(data))
            assert caught.value.kind.value == "llm_invalid_response"


@pytest.mark.parametrize("lines,valid", [(4096, True), (4097, False)])
async def test_event_line_limit(lines, valid):
    async with asyncio.timeout(5):
        decoder = SSEDecoder()
        if valid:
            assert list(decoder.feed(b":\n" * lines + b"\n")) == []
        else:
            with pytest.raises(LLMError):
                list(decoder.feed(b":\n" * lines + b"\n"))


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"choices": "bad"},
        {"choices": [None]},
        {"choices": [{"message": {"content": 4}}]},
        {"choices": [{"message": {"tool_calls": "bad"}}]},
    ],
)
async def test_nonstream_envelope_is_terminal(body):
    async with asyncio.timeout(5):
        with pytest.raises(LLMError) as caught:
            LLMClient._parse_response(body)
        assert caught.value.kind.value == "llm_invalid_response"


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", None])
async def test_functions_fragments_keep_one_identity_and_done_tool_only(finish_reason):
    async with asyncio.timeout(5):
        model = replace(_model(), dialect="functions")
        body = (
            _sse(
                {"choices": [{"delta": {"function_call": {"name": "read"}}}]},
                {
                    "choices": [
                        {"delta": {"function_call": {"name": "read", "arguments": '{"path":'}}}
                    ]
                },
                {"choices": [{"delta": {"function_call": {"arguments": '"a"}'}}}]},
                {"choices": [{"delta": {}, "finish_reason": finish_reason}]},
            )
            + b"data: [DONE]\n\n"
        )
        client = _client(model, lambda _: httpx.Response(200, content=body))

        async def delta(kind, text):
            pytest.fail("tool-only reply published text")

        result = await client.send(_request(model, stream=True), on_delta=delta)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"]
        assert result.tool_calls[0]["function"] == {"name": "read", "arguments": '{"path":"a"}'}
        await client.aclose()


@pytest.mark.parametrize(
    "bad",
    [
        {"choices": [None]},
        {"choices": [{"delta": 1}]},
        {"choices": [{"delta": {"content": []}}]},
        {"choices": [{"delta": {"reasoning_content": 4}}]},
        {"choices": [{"delta": {"tool_calls": {}}}]},
        *[
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": value, "function": {"name": "read"}}]}}
                ]
            }
            for value in [True, -1, "0", None]
        ],
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "other"}]}}]},
    ],
)
async def test_invalid_stream_structure(bad):
    async with asyncio.timeout(5):
        with pytest.raises(LLMError) as caught:
            ChatStreamAccumulator().add(json.dumps(bad))
        assert caught.value.kind.value == "llm_invalid_response"


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", None),
        ("arguments", ""),
        ("arguments", "[1]"),
        ("arguments", "null"),
        ("arguments", "broken"),
        ("arguments", None),
    ],
)
async def test_one_invalid_call_rejects_entire_nonstream_reply(field, value):
    async with asyncio.timeout(5):
        good = {"type": "function", "function": {"name": "read", "arguments": "{}"}}
        bad = {"type": "function", "function": {"name": "read", "arguments": "{}", field: value}}
        with pytest.raises(LLMError) as caught:
            LLMClient._parse_response(
                {
                    "choices": [
                        {
                            "message": {"content": "answer", "tool_calls": [good, bad]},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        assert caught.value.kind.value == "llm_invalid_response"


async def test_conflicting_finish_and_metadata_are_invalid():
    async with asyncio.timeout(5):
        acc = ChatStreamAccumulator()
        add(acc, {"content": "answer"}, finish_reason="stop")
        with pytest.raises(LLMError):
            add(acc, finish_reason="length")
        acc = ChatStreamAccumulator()
        add(
            acc,
            {
                "tool_calls": [
                    {"index": 0, "id": "one", "function": {"name": "read", "arguments": "{"}}
                ]
            },
        )
        with pytest.raises(LLMError):
            add(acc, {"tool_calls": [{"index": 0, "id": "two"}]})
        with pytest.raises(LLMError):
            add(acc, {"tool_calls": [{"index": 0, "function": {"name": "write"}}]})


@pytest.mark.parametrize("finish_reason", ["content_filter", "tool_calls"])
async def test_terminal_finish_must_match_usable_response(finish_reason):
    async with asyncio.timeout(5):
        with pytest.raises(LLMError) as caught:
            LLMClient._parse_response(
                {"choices": [{"message": {"content": "answer"}, "finish_reason": finish_reason}]}
            )
        assert caught.value.kind.value == "llm_invalid_response"


async def test_missing_duplicate_ids_are_replaced_after_validation():
    async with asyncio.timeout(5):
        calls = [
            {"id": value, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            for value in [None, "same", "same"]
        ]
        result = LLMClient._parse_response({"choices": [{"message": {"tool_calls": calls}}]})
        assert len({call["id"] for call in result.tool_calls}) == 3


async def test_huge_line_without_newline_is_bounded_and_many_events_are_lazy():
    async with asyncio.timeout(5):
        decoder = SSEDecoder()
        with pytest.raises(LLMError):
            list(decoder.iter_feed(b"x" * 262145))
        decoder = SSEDecoder()
        events = decoder.iter_feed(b"data: first\n\n" + b"x" * 300000)
        assert next(events) == "first"
        # The oversized tail has not been read or validated.
        with pytest.raises(LLMError):
            next(events)


@pytest.mark.parametrize("mode", ["separate_channels", "arguments", "tagged_hit"])
async def test_repetition_channels_tags_and_excluded_arguments(mode):
    async with asyncio.timeout(5):
        from agent.model_config import ReasoningSpec, ReasoningTagsSpec

        block = "".join(chr(1024 + i) for i in range(32))
        model = replace(
            _model(max_retries=3),
            repetition_threshold_chars=4096,
            reasoning=ReasoningSpec(tags=ReasoningTagsSpec(start="<think>", end="</think>")),
        )
        if mode == "arguments":
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {
                                            "name": "write",
                                            "arguments": json.dumps(
                                                {"text": block * 128}, ensure_ascii=False
                                            ),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ]
        elif mode == "separate_channels":
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {"content": block * 64, "reasoning_content": block * 64},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ]
        else:
            chunks = [
                {"choices": [{"delta": {"content": text}}]}
                for text in ["<thi", "nk>" + block * 64, block * 64 + "</think>answer"]
            ]
        calls, published = [], []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, content=_sse(*chunks) + b"data: [DONE]\n\n")

        client = _client(model, handler)

        async def delta(kind, text):
            published.append((kind, text))

        try:
            if mode == "tagged_hit":
                with pytest.raises(LLMError) as caught:
                    await client.send(_request(model, stream=True), on_delta=delta)
                assert caught.value.kind.value == "llm_repetition"
                assert sum(len(text) for kind, text in published if kind == "reasoning") < 4096
            else:
                await client.send(_request(model, stream=True), on_delta=delta)
            assert len(calls) == 1
        finally:
            await client.aclose()


@pytest.mark.parametrize("started", [False, True])
async def test_incomplete_has_exact_started_and_bounded_http_retries(started, monkeypatch):
    async with asyncio.timeout(5):
        calls = []
        model = _model(max_retries=2)

        def handler(request):
            calls.append(request)
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "choices": [
                            {"delta": {"content": "partial"} if started else {"role": "assistant"}}
                        ]
                    }
                ),
            )

        client = _client(model, handler)
        real_sleep = __import__("asyncio").sleep

        async def backoff(seconds):
            await real_sleep(0)

        monkeypatch.setattr("agent.llm.asyncio.sleep", backoff)

        async def delta(kind, text):
            pass

        with pytest.raises(LLMError) as caught:
            await client.send(_request(model, stream=True), on_delta=delta)
        assert caught.value.kind.value == "llm_stream_incomplete"
        assert caught.value.started is started
        assert len(calls) == (1 if started else 3)
        await client.aclose()


async def test_empty_tool_delta_is_not_a_started_reply():
    async with asyncio.timeout(5):
        acc = ChatStreamAccumulator()
        add(acc, {"tool_calls": [{"index": 0, "type": "function", "function": {}}]})
        with pytest.raises(LLMError) as caught:
            acc.finalize()
        assert caught.value.kind.value == "llm_stream_incomplete"
        assert not caught.value.started


@pytest.mark.parametrize("size,valid", [(262144, True), (262145, False)])
async def test_crlf_bytes_count_toward_event_limit(size, valid):
    async with asyncio.timeout(5):
        data = b":" + b"x" * (size - 3) + b"\r\n\r\n"
        decoder = SSEDecoder()
        if valid:
            assert list(decoder.feed(data)) == []
        else:
            with pytest.raises(LLMError):
                list(decoder.feed(data))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["stop", "content_filter", 17])
async def test_guard_terminal_audit_preserves_reported_usage(stream, reason):

    async with asyncio.timeout(5):
        body = {
            "choices": [
                {"delta" if stream else "message": {"content": " "}, "finish_reason": reason}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        client = _client(
            _model(),
            lambda _: (
                httpx.Response(200, content=_sse(body))
                if stream
                else httpx.Response(200, json=body)
            ),
        )
        records = []
        terminal = asyncio.Event()

        async def attempt(record):
            records.append(record)
            if record.status != "started":
                terminal.set()

        async def delta(kind, text):
            pass

        with pytest.raises(LLMError) as caught:
            await client.send(
                _request(client.model, stream=stream),
                on_delta=delta if stream else None,
                on_attempt=attempt,
            )
        assert caught.value.kind.value == (
            "llm_empty_response" if reason == "stop" else "llm_invalid_response"
        )
        if reason == "content_filter":
            assert "фильтр" in str(caught.value)
        await terminal.wait()
        assert [(r.prompt_tokens, r.completion_tokens) for r in records if r.status == "error"] == [
            (7, 3)
        ]
        await client.aclose()


async def test_repetition_discards_text_before_accumulation_and_publication(monkeypatch):
    async with asyncio.timeout(5):
        from agent import sse

        snapshots = []
        original = sse.ChatStreamAccumulator

        class Accumulator(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                snapshots.append(self)

        monkeypatch.setattr(sse, "ChatStreamAccumulator", Accumulator)
        model = replace(_model(), repetition_threshold_chars=4096)
        body = _sse(*[{"choices": [{"delta": {"content": "z" * 2048}}]}] * 2)
        client = _client(model, lambda _: httpx.Response(200, content=body))
        published = []

        async def delta(kind, text):
            published.append(text)

        with pytest.raises(LLMError) as caught:
            await client.send(_request(model, stream=True), on_delta=delta)
        assert caught.value.kind.value == "llm_repetition"
        assert "".join(published) == "z" * 2048
        assert snapshots[0].result().content == "z" * 2048
        await client.aclose()


@pytest.mark.parametrize("mode", ["missing_type", "conflicting_state", "wrong_choice"])
async def test_original_stream_metadata_is_not_repaired(mode):
    async with asyncio.timeout(5):
        model = _model()
        call = {"index": 0, "function": {"name": "read", "arguments": "{}"}}
        if mode != "missing_type":
            call.update(type="function", functions_state_id="first")
        chunks = [{"choices": [{"delta": {"tool_calls": [call]}}]}]
        if mode == "conflicting_state":
            chunks.append(
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": 0, "functions_state_id": "second"}]}}
                    ]
                }
            )
        if mode == "wrong_choice":
            chunks[0]["choices"].append(17)
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        client = _client(model, lambda _: httpx.Response(200, content=_sse(*chunks)))

        async def delta(kind, text):
            pass

        try:
            with pytest.raises(LLMError) as caught:
                await client.send(_request(model, stream=True), on_delta=delta)
            assert caught.value.kind.value == "llm_invalid_response"
        finally:
            await client.aclose()
