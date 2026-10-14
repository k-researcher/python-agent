"""Unit tests for the context compression coordinator (stage 2.3c)."""

from __future__ import annotations

import asyncio
import json
from math import ceil
from typing import Any

import pytest

from agent.context_manager import (
    ContextBudgetError,
    ContextManager,
    ContextResult,
)
from agent.context_projection import (
    HistorySnapshot,
    SourceMessage,
    chain_hash,
    project,
)
from agent.context_summary import (
    SUMMARY_LABEL,
    SUMMARY_VERSION,
    SummaryCheckpoint,
)
from agent.model_config import ReasoningSpec, ResolvedModel

VALID_HASH = "b" * 64


def model(model_id: str = "m", *, window: int = 4000, max_tokens: int = 200) -> ResolvedModel:
    return ResolvedModel(
        id=model_id,
        provider_id="p",
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test-secret",
        configured=True,
        model=model_id,
        context_window=window,
        max_tokens=max_tokens,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
    )


def sm(
    id: int,
    role: str,
    content: str | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> SourceMessage:
    return SourceMessage(
        id=id,
        role=role,
        kind="normal",
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def exchange(
    id_start: int,
    *,
    big_chars: int = 0,
    assistant_content: str = "assistant text",
    big_write: int = 0,
) -> list[SourceMessage]:
    call_id = f"call_{id_start}"
    assistant = sm(
        id_start,
        "assistant",
        assistant_content,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": "/tmp/f", "content": "W" * big_write}),
                },
            }
        ],
    )
    tool = sm(
        id_start + 1,
        "tool",
        json.dumps({"success": True, "output": "Z" * big_chars}),
        tool_call_id=call_id,
    )
    return [assistant, tool]


def big_snapshot() -> HistorySnapshot:
    """Three compressible exchanges between a protected first/last user."""
    messages: list[SourceMessage] = [
        sm(1, "system", "You are a helpful agent."),
        sm(2, "user", "task one"),
    ]
    messages += exchange(3, big_chars=3000)
    messages += exchange(5, big_chars=3000)
    messages += exchange(7, big_chars=3000)
    messages.append(sm(9, "user", "continue please"))
    messages += exchange(10, big_chars=3000)
    messages += exchange(12, big_chars=3000)
    return HistorySnapshot(tuple(messages), None, None)


def stage2_snapshot() -> HistorySnapshot:
    """Compressible exchanges with large write arguments (folding helps)."""
    messages: list[SourceMessage] = [
        sm(1, "system", "You are a helpful agent."),
        sm(2, "user", "task one"),
    ]
    messages += exchange(3, big_chars=4000, big_write=2000)
    messages += exchange(5, big_chars=4000, big_write=2000)
    messages += exchange(7, big_chars=4000, big_write=2000)
    messages.append(sm(9, "user", "continue please"))
    messages += exchange(10, big_chars=3000)
    messages += exchange(12, big_chars=3000)
    return HistorySnapshot(tuple(messages), None, None)


def checkpoint(text: str = "svm summary", covers: int = 2) -> SummaryCheckpoint:
    return SummaryCheckpoint(
        text=text,
        covers_until=covers,
        source_hash=VALID_HASH,
        version=SUMMARY_VERSION,
    )


async def run(
    snapshot: HistorySnapshot,
    *,
    manager_tokens: int = 0,
    mdl: ResolvedModel | None = None,
    tools: list[dict[str, Any]] | None = None,
    runner: Any = None,
    **kwargs: Any,
) -> ContextResult:
    mgr = ContextManager(manager_tokens)
    return await mgr.prepare(
        snapshot,
        mdl or model(),
        tools or [],
        summary_runner=runner,
        **kwargs,
    )


async def test_context_fits_stage_none_unchanged() -> None:
    messages: list[SourceMessage] = [
        sm(1, "system", "You are an agent."),
        sm(2, "user", "hello"),
        sm(3, "assistant", "hi"),
        sm(4, "user", "thanks"),
    ]
    snapshot = HistorySnapshot(tuple(messages), None, None)
    result = await run(snapshot, mdl=model(window=16000, max_tokens=500))
    assert result.stage == "none"
    assert result.fits is True
    assert result.masked_tool_results == 0
    assert result.folded_tool_calls == 0
    assert result.checkpoint_to_save is None
    assert result.messages == result.request.payload["messages"]

    roles = [item["role"] for item in result.messages]
    assert roles == ["system", "user", "assistant", "user"]


async def test_tools_increase_raw_tokens() -> None:
    snapshot = HistorySnapshot((sm(1, "user", "hi"),), None, None)
    tools = [
        {
            "type": "function",
            "function": {"name": "shell", "parameters": {"type": "object"}},
        }
    ]
    without = await run(snapshot, mdl=model(window=16000, max_tokens=500))
    with_tools = await run(
        snapshot,
        mdl=model(window=16000, max_tokens=500),
        tools=tools,
    )
    assert with_tools.raw_tokens > without.raw_tokens


async def test_calibration_factor_applied_once() -> None:
    snapshot = HistorySnapshot((sm(1, "user", "some content here"),), None, None)
    result = await run(
        snapshot,
        mdl=model(window=16000, max_tokens=500),
        calibration_factor=2.0,
    )
    assert result.approximate_tokens == result.raw_tokens * 2


async def test_larger_effective_max_tokens_reduces_target() -> None:
    snapshot = HistorySnapshot((sm(1, "user", "hi"),), None, None)
    low = await run(
        snapshot,
        mdl=model(window=16000, max_tokens=200),
        effective_max_tokens=100,
    )
    high = await run(
        snapshot,
        mdl=model(window=16000, max_tokens=200),
        effective_max_tokens=800,
    )
    assert high.target_tokens < low.target_tokens
    assert high.effective_max_tokens == 800
    assert low.effective_max_tokens == 100


async def test_tiny_window_invalid_budget() -> None:
    snapshot = HistorySnapshot((sm(1, "user", "hi"),), None, None)
    with pytest.raises(ContextBudgetError) as exc:
        await run(snapshot, mdl=model(window=64, max_tokens=1000))
    assert exc.value.reason == "invalid_budget"


def _content(result: ContextResult) -> list[dict[str, Any]]:
    return result.messages


async def test_protected_blocks_unchanged_on_all_stages() -> None:
    snapshot = big_snapshot()

    result1 = await run(
        snapshot,
        mdl=model(),
        target_tokens=2400,
    )
    assert result1.stage == "stage_1"
    msgs1 = _content(result1)
    assert msgs1[0] == {"role": "system", "content": "You are a helpful agent."}
    assert msgs1[1] == {"role": "user", "content": "task one"}

    last_user = next(item for item in reversed(msgs1) if item["role"] == "user")
    assert last_user == {"role": "user", "content": "continue please"}

    tail_tools = [
        item
        for item in msgs1
        if item.get("role") == "tool" and ("Z" * 3000) in (item.get("content") or "")
    ]
    assert len(tail_tools) == 2


async def test_big_exchange_not_split() -> None:

    snapshot = stage2_snapshot()
    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=2500,
    )
    assert result.stage == "stage_2"
    for index, item in enumerate(result.messages):
        calls = item.get("tool_calls")
        if not calls:
            continue
        next_item = result.messages[index + 1]
        assert next_item["role"] == "tool"
        assert next_item["tool_call_id"] == calls[0]["id"]


async def test_stops_at_first_fitting_stage() -> None:

    snapshot = big_snapshot()
    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=2200,
    )
    assert result.stage == "stage_1"
    assert result.masked_tool_results == 3


async def test_stage_2_does_not_remask() -> None:
    snapshot = stage2_snapshot()
    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=2500,
    )
    assert result.stage == "stage_2"
    assert result.masked_tool_results == 3
    assert result.folded_tool_calls == 3
    for item in result.messages:
        content = item.get("content")
        if isinstance(content, str) and "masked" in content:
            parsed = json.loads(content)
            assert parsed.get("masked") is True
            assert "Z" * 1 in (parsed.get("excerpt") or "")
            assert not isinstance(parsed.get("output"), str)


async def test_expanding_masking_rejected() -> None:

    messages: list[SourceMessage] = [
        sm(1, "system", "sys"),
        sm(2, "user", "task"),
        *exchange(3, assistant_content="a", big_chars=0),
        *exchange(5, assistant_content="b", big_chars=5000),
        *exchange(7, assistant_content="c", big_chars=5000),
        *exchange(9, assistant_content="d", big_chars=5000),
    ]
    snapshot = HistorySnapshot(tuple(messages), None, None)
    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=3200,
    )
    assert result.stage == "stage_1"
    assert result.masked_tool_results == 1
    tiny_token = json.dumps({"success": True, "output": ""})
    tiny_tool = next(item for item in result.messages if item.get("tool_call_id") == "call_3")
    assert tiny_tool["content"] == tiny_token


async def test_protected_minimum_over_budget_runner_not_called() -> None:
    snapshot = big_snapshot()
    calls: list[list[dict[str, Any]]] = []

    async def runner(items: list[dict[str, Any]]) -> str:
        calls.append(items)
        return "## Goal\n- x\n"

    with pytest.raises(ContextBudgetError) as exc:
        await run(
            snapshot,
            mdl=model(),
            target_tokens=300,
            runner=runner,
            summary_model_id="light",
            summary_input_tokens=100000,
        )
    assert exc.value.reason == "protected_over_budget"
    assert calls == []


def _summary_like(result: ContextResult) -> dict[str, Any]:
    return next(
        (
            item
            for item in result.messages
            if isinstance(item.get("content"), str) and SUMMARY_LABEL in item["content"]
        ),
        {},
    )


async def test_summary_placement_and_counts() -> None:
    snapshot = big_snapshot()

    async def runner(items: list[dict[str, Any]]) -> str:
        return (
            "## Goal\n- g\n"
            "## Constraints\n- c\n"
            "## Decisions\n- d\n"
            "## Changed paths\n- /a\n"
            "## Check results\n- ok\n"
            "## Open tasks\n- t\n"
        )

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=runner,
        summary_model_id="light",
        summary_input_tokens=100000,
    )
    assert result.stage == "summary"
    assert result.summary_used is True
    assert result.summary_calls == 1
    assert result.checkpoint_model == "light"
    assert result.summary_used

    item = _summary_like(result)
    assert item["role"] == "user"
    idx = result.messages.index(item)
    assert result.messages[idx - 1] == {
        "role": "user",
        "content": "task one",
    }

    assert "Z" * 3000 not in "".join(m.get("content") or "" for m in result.messages[: idx + 1])
    assert result.would_summarize is False
    assert result.omitted_messages >= 6


async def test_covers_until_is_max_prefix_id() -> None:
    snapshot = big_snapshot()

    async def runner(items: list[dict[str, Any]]) -> str:
        return (
            "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
            "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
        )

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=runner,
        summary_model_id="light",
        summary_input_tokens=100000,
    )
    assert result.checkpoint_to_save is not None
    assert result.checkpoint_to_save.covers_until == 8


async def test_chain_hash_independent_of_masking() -> None:
    snapshot = big_snapshot()

    async def runner(items: list[dict[str, Any]]) -> str:
        return (
            "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
            "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
        )

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=runner,
        summary_model_id="light",
        summary_input_tokens=100000,
    )
    checkpoint = result.checkpoint_to_save
    assert checkpoint is not None

    proj = project(big_snapshot())
    prefix_id_items = [
        {"id": source_id, **item}
        for item, source_id in zip(proj.items, proj.source_ids, strict=True)
        if source_id in {3, 4, 5, 6, 7, 8}
    ]
    assert checkpoint.source_hash == chain_hash(None, prefix_id_items)


async def test_runner_exception_propagates() -> None:

    snapshot = big_snapshot()

    async def failing(items: list[dict[str, Any]]) -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run(
            snapshot,
            mdl=model(),
            target_tokens=1800,
            runner=failing,
            summary_model_id="light",
            summary_input_tokens=100000,
        )


async def test_runner_returns_none_leads_to_deterministic() -> None:
    snapshot = big_snapshot()

    async def none_runner(items: list[dict[str, Any]]) -> None:
        return None

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=none_runner,
        summary_input_tokens=100000,
    )
    assert result.stage == "deterministic"
    assert result.summary_calls == 1


async def test_cancelled_error_propagates() -> None:
    snapshot = big_snapshot()

    async def cancel_runner(items: list[dict[str, Any]]) -> str:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run(
            snapshot,
            mdl=model(),
            target_tokens=1800,
            runner=cancel_runner,
            summary_input_tokens=100000,
        )


async def test_existing_checkpoint_used_in_projection_and_earlier_summary() -> None:
    cp = checkpoint(text="PREVIOUS SUMMARY TEXT", covers=2)
    snapshot = HistorySnapshot(big_snapshot().messages, cp, checkpoint_message_id=99)

    result = await run(snapshot, mdl=model(), target_tokens=1800)
    assert result.stage == "deterministic"
    assert result.summary_used is True
    assert result.checkpoint_message_id == 99
    item = _summary_like(result)
    assert "PREVIOUS SUMMARY TEXT" in item["content"]
    assert result.checkpoint_to_save is not None
    assert "Earlier summary:\nPREVIOUS SUMMARY TEXT" in (result.checkpoint_to_save.text)


async def test_would_summarize_without_runner() -> None:

    result = await run(
        big_snapshot(),
        mdl=model(),
        target_tokens=1800,
    )
    assert result.stage == "deterministic"
    assert result.would_summarize is True
    assert result.checkpoint_model is None
    assert result.summary_used is True


async def test_light_wire_trims_paid_prefix_never_over_budget() -> None:
    from agent.summary_runner import summary_effort
    from agent.wire import build_wire_request as bwr
    from agent.wire import estimate_wire as ew

    def sent_wire(items: list[dict[str, Any]]) -> int:
        request = bwr(
            light,
            items,
            [],
            reasoning_effort=summary_effort(light),
            max_tokens=min(1024, light.max_tokens),
            stream=False,
        )
        return ceil(ew(request) * 2.0)

    snapshot = big_snapshot()
    light = model("light", window=2000, max_tokens=200)
    outcomes: dict[int, tuple[ContextResult, list[list[dict[str, Any]]]]] = {}

    for budget in (100, 3000, 6000):
        seen: list[list[dict[str, Any]]] = []

        async def runner(
            items: list[dict[str, Any]], _seen: list[list[dict[str, Any]]] = seen
        ) -> str:
            _seen.append(items)
            return (
                "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
                "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
            )

        try:
            result = await run(
                snapshot,
                mdl=model(),
                target_tokens=1800,
                runner=runner,
                summary_model=light,
                summary_input_tokens=budget,
                summary_calibration_factor=2.0,
            )
        except ContextBudgetError as exc:
            outcomes[budget] = (exc.result, seen)
        else:
            outcomes[budget] = (result, seen)

    _, seen_tiny = outcomes[100]
    assert seen_tiny == []

    _, seen_reduced = outcomes[3000]
    assert seen_reduced
    assert sent_wire(seen_reduced[0]) <= 3000

    full_result, seen_full = outcomes[6000]
    assert full_result.stage == "summary"
    assert seen_full
    assert sent_wire(seen_full[0]) <= 6000
    assert full_result.checkpoint_to_save is not None
    assert full_result.checkpoint_to_save.covers_until == 8


async def test_repeat_prepare_is_idempotent_and_does_not_mutate_input() -> None:
    snapshot = big_snapshot()
    original_ids = [message.id for message in snapshot.messages]
    first = await run(snapshot, mdl=model(), target_tokens=1800)
    second = await run(snapshot, mdl=model(), target_tokens=1800)
    assert first.stage == second.stage == "deterministic"
    assert first.messages == second.messages

    assert [message.id for message in snapshot.messages] == original_ids


async def test_summary_preserves_protected_tail_unchanged() -> None:
    snapshot = big_snapshot()

    async def runner(items: list[dict[str, Any]]) -> str:
        return (
            "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
            "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
        )

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=runner,
        summary_model_id="light",
        summary_input_tokens=100000,
    )
    assert result.stage == "summary"
    msgs = result.messages

    assert msgs[0] == {"role": "system", "content": "You are a helpful agent."}

    assert {"role": "user", "content": "continue please"} in msgs

    big_tools = [
        item.get("content", "")
        for item in msgs
        if item.get("role") == "tool" and "Z" * 3000 in item.get("content", "")
    ]
    assert len(big_tools) == 2
    for content in big_tools:
        assert "masked" not in content

    assert result.masked_tool_results == 0
    assert result.folded_tool_calls == 0


async def test_deterministic_preserves_protected_tail_unchanged() -> None:
    snapshot = big_snapshot()
    result = await run(snapshot, mdl=model(), target_tokens=1800)
    assert result.stage == "deterministic"
    msgs = result.messages
    assert msgs[0] == {"role": "system", "content": "You are a helpful agent."}
    assert {"role": "user", "content": "continue please"} in msgs
    big_tools = [
        item.get("content", "")
        for item in msgs
        if item.get("role") == "tool" and "Z" * 3000 in item.get("content", "")
    ]
    assert len(big_tools) == 2
    for content in big_tools:
        assert "masked" not in content
    assert result.masked_tool_results == 0
    assert result.folded_tool_calls == 0


async def test_summary_preserves_unfinished_exchange() -> None:

    messages: list[SourceMessage] = [
        sm(1, "system", "sys"),
        sm(2, "user", "task"),
        *exchange(3, big_chars=4000),
        *exchange(5, big_chars=4000),
        sm(7, "user", "continue"),
        *exchange(8, big_chars=4000),
        sm(
            10,
            "assistant",
            "thinking",
            tool_calls=[
                {
                    "id": "call_90",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({"command": "ls"}),
                    },
                }
            ],
        ),
    ]
    snapshot = HistorySnapshot(tuple(messages), None, None)

    async def runner(items: list[dict[str, Any]]) -> str:
        return (
            "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
            "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
        )

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=2350,
        runner=runner,
        summary_model_id="light",
        summary_input_tokens=100000,
    )
    assert result.stage == "summary"
    incomplete = next(
        item
        for item in result.messages
        if item.get("tool_call_id") == "call_90"
        or (item.get("tool_calls") and item["tool_calls"][0]["id"] == "call_90")
    )
    assert incomplete["content"] == "thinking"

    tool_replies = [i for i in result.messages if i.get("role") == "tool"]
    assert all(t.get("tool_call_id") != "call_90" for t in tool_replies)


async def test_error_results_report_fits_false() -> None:

    with pytest.raises(ContextBudgetError) as exc:
        await run(
            HistorySnapshot((sm(1, "user", "hi"),), None, None),
            mdl=model(window=64, max_tokens=1000),
        )
    assert exc.value.reason == "invalid_budget"
    assert exc.value.result.fits is False
    assert exc.value.result.approximate_tokens > 0

    with pytest.raises(ContextBudgetError) as exc:
        await run(big_snapshot(), mdl=model(), target_tokens=300)
    assert exc.value.reason == "protected_over_budget"
    assert exc.value.result.fits is False


async def test_returned_stage_strictly_reduces_raw() -> None:

    snapshot = big_snapshot()
    none_r = await run(
        snapshot,
        mdl=model(window=64000, max_tokens=1000),
        target_tokens=100000,
    )
    assert none_r.stage == "none"
    none_raw = none_r.raw_tokens
    stage1 = await run(snapshot, mdl=model(), target_tokens=2200)
    assert stage1.stage == "stage_1"
    assert stage1.masked_tool_results > 0
    assert stage1.raw_tokens < none_raw

    s2_snapshot = stage2_snapshot()
    none_s2 = await run(
        s2_snapshot,
        mdl=model(window=64000, max_tokens=1000),
        target_tokens=100000,
    )
    assert none_s2.stage == "none"
    stage2 = await run(s2_snapshot, mdl=model(), target_tokens=2500)
    assert stage2.stage == "stage_2"
    assert stage2.folded_tool_calls > 0
    assert stage2.raw_tokens < none_s2.raw_tokens


async def test_long_old_checkpoint_not_truncated() -> None:

    marker = "UNIQUE_END_MARKER_" + "z" * 200
    old = SummaryCheckpoint(
        text="OLD " + "o" * 600 + "\n" + marker,
        covers_until=2,
        source_hash="h",
        version=SUMMARY_VERSION,
    )
    snapshot = HistorySnapshot(big_snapshot().messages, old, checkpoint_message_id=99)
    result = await run(snapshot, mdl=model(), target_tokens=2200)
    assert result.stage == "deterministic"
    assert result.checkpoint_to_save is not None
    assert marker in result.checkpoint_to_save.text
    assert "OLD " in result.checkpoint_to_save.text


async def test_over_output_budget_summary_rejected() -> None:
    snapshot = big_snapshot()
    light = model("light", window=2000, max_tokens=200)

    async def huge_runner(items: list[dict[str, Any]]) -> str:
        return "## Goal\n" + "x" * 5000 + "\n## Constraints\n- c\n"

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=huge_runner,
        summary_model=light,
        summary_model_id="light",
        summary_input_tokens=100000,
    )

    assert result.stage == "deterministic"
    assert result.summary_calls == 1
    assert result.checkpoint_model is None


async def test_blank_runner_output_rejected_before_normalize() -> None:
    snapshot = big_snapshot()

    async def blank_runner(items: list[dict[str, Any]]) -> str:
        return "   \n\t  "

    result = await run(
        snapshot,
        mdl=model(),
        target_tokens=1800,
        runner=blank_runner,
        summary_input_tokens=100000,
    )
    assert result.stage == "deterministic"
    assert result.summary_calls == 1
    assert result.checkpoint_model is None


async def test_multi_turn_summarizes_across_middle_user() -> None:
    """A long user/assistant conversation must summarize across a middle user,
    preserving the first/last true user and the last two completed blocks."""

    def multi_turn() -> HistorySnapshot:
        messages: list[SourceMessage] = [
            sm(1, "system", "sys"),
            sm(2, "user", "first task"),
            sm(3, "assistant", "A" * 2000),
            sm(4, "user", "middle question 1"),
            sm(5, "assistant", "B" * 2000),
            sm(6, "user", "middle question 2"),
            sm(7, "assistant", "C" * 2000),
            sm(8, "user", "continue"),
        ]
        messages += exchange(9, big_chars=2000)
        messages += exchange(11, big_chars=2000)
        return HistorySnapshot(tuple(messages), None, None)

    async def runner(items: list[dict[str, Any]]) -> str:
        return (
            "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
            "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
        )

    snap = multi_turn()
    for paid in (True, False):
        result = await run(
            snap,
            mdl=model(),
            target_tokens=1800,
            runner=runner if paid else None,
            summary_model_id="light",
            summary_input_tokens=100000,
        )
        assert result.stage == ("summary" if paid else "deterministic")
        assert result.checkpoint_to_save is not None

        assert result.checkpoint_to_save.covers_until == 7
        assert result.omitted_messages == 5
        msgs = result.messages

        assert msgs[0] == {"role": "system", "content": "sys"}
        contents = {item.get("content") for item in msgs}

        assert "first task" in contents
        assert "continue" in contents

        assert "middle question 1" not in contents
        assert "middle question 2" not in contents
        assert "A" * 2000 not in contents
        assert "B" * 2000 not in contents
        assert "C" * 2000 not in contents

        summary_items = [m for m in msgs if m.get("content", "").startswith(SUMMARY_LABEL)]
        assert len(summary_items) == 1

        big_tools = [m.get("content", "") for m in msgs if "Z" * 2000 in m.get("content", "")]
        assert len(big_tools) == 2


async def test_light_budget_backs_up_past_middle_user() -> None:
    """A light-model budget cut right after a middle user must back up to the
    previous completed non-user boundary and keep that user in the tail."""

    def multi_turn() -> HistorySnapshot:
        messages: list[SourceMessage] = [
            sm(1, "system", "sys"),
            sm(2, "user", "first task"),
            sm(3, "assistant", "A" * 2000),
            sm(4, "user", "middle question 1"),
            sm(5, "assistant", "B" * 2000),
            sm(6, "user", "middle question 2"),
            sm(7, "assistant", "C" * 2000),
            sm(8, "user", "continue"),
        ]
        messages += exchange(9, big_chars=2000)
        messages += exchange(11, big_chars=2000)
        return HistorySnapshot(tuple(messages), None, None)

    snap = multi_turn()
    seen: list[list[dict[str, Any]]] = []

    async def runner(items: list[dict[str, Any]]) -> str:
        seen.append(items)
        return (
            "## Goal\n- g\n## Constraints\n- c\n## Decisions\n- d\n"
            "## Changed paths\n- /a\n## Check results\n- ok\n## Open tasks\n- t\n"
        )

    light = model("light", window=2000, max_tokens=200)
    result = await run(
        snap,
        mdl=model(),
        target_tokens=1800,
        runner=runner,
        summary_model=light,
        summary_model_id="light",
        summary_input_tokens=2600,
        summary_calibration_factor=2.0,
    )
    assert result.stage == "summary"
    assert seen
    assert len(seen) == 1
    assert result.checkpoint_to_save is not None

    assert result.checkpoint_to_save.covers_until == 5
    contents = {item.get("content") for item in result.messages}

    assert "middle question 2" in contents

    assert "middle question 1" not in contents

    assert "first task" in contents
    assert "continue" in contents


async def test_deterministic_text_keeps_previous_in_full_within_budget() -> None:
    from agent.context_manager import _det_tokens, _fit_deterministic_text
    from agent.context_summary import SUMMARY_MAX_TOKENS

    marker = "\nMARKER_TAIL"
    previous = "Earlier " + "P" * 1500 + marker
    new = "\n".join(f"- record {i} " + "x" * 200 for i in range(30))

    result = _fit_deterministic_text(previous, new)
    assert result is not None

    assert _det_tokens(result) <= SUMMARY_MAX_TOKENS

    assert marker in result
    assert previous in result

    for i in range(30):
        if f"- record {i} " in result:
            assert f"- record {i} " + "x" * 200 in result

    huge_previous = "H" * (SUMMARY_MAX_TOKENS * 4 + 100)
    assert _fit_deterministic_text(huge_previous, "## Goal\n- g") is None


async def test_preview_does_not_offer_summary_when_light_prefix_is_empty() -> None:
    manager = ContextManager(reserved_tokens=1024)
    with pytest.raises(ContextBudgetError) as exc:
        await manager.prepare(
            big_snapshot(),
            model(),
            [],
            target_tokens=1750,
            summary_runner=None,
            summary_input_tokens=1,
        )
    assert exc.value.reason == "no_fit"
    assert exc.value.result.would_summarize is False


def test_deterministic_text_declines_when_no_new_record_fits() -> None:
    from agent.context_manager import _det_tokens, _fit_deterministic_text, _join_det
    from agent.context_summary import SUMMARY_MAX_TOKENS

    overhead = len(_join_det("", ""))
    previous = "P" * (SUMMARY_MAX_TOKENS * 4 - overhead)
    assert _det_tokens(_join_det(previous, "")) == SUMMARY_MAX_TOKENS
    assert _fit_deterministic_text(previous, "- new record") is None


async def test_protected_over_budget_would_summarize_false() -> None:
    with pytest.raises(ContextBudgetError) as exc:
        await run(big_snapshot(), mdl=model(), target_tokens=300)
    assert exc.value.reason == "protected_over_budget"
    assert exc.value.result.would_summarize is False


async def test_no_fit_empty_prefix_would_summarize_false() -> None:
    call_id = "call_90"
    messages: list[SourceMessage] = [
        sm(1, "system", "sys"),
        sm(2, "user", "first task"),
        sm(
            3,
            "assistant",
            "thinking",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({"command": "ls"}),
                    },
                }
            ],
        ),
    ]
    messages += exchange(5, big_chars=2000)
    messages += exchange(7, big_chars=2000)
    messages += exchange(9, big_chars=2000)
    messages.append(sm(11, "user", "last request"))
    messages += exchange(12, big_chars=3000)
    messages += exchange(14, big_chars=3000)
    snapshot = HistorySnapshot(tuple(messages), None, None)

    with pytest.raises(ContextBudgetError) as exc:
        await run(snapshot, mdl=model(), target_tokens=1800)
    assert exc.value.reason == "no_fit"
    assert exc.value.result.would_summarize is False


async def test_no_fit_nonempty_prefix_without_runner_would_summarize_true() -> None:
    with pytest.raises(ContextBudgetError) as exc:
        await run(big_snapshot(), mdl=model(), target_tokens=1750)
    assert exc.value.reason == "no_fit"
    assert exc.value.result.would_summarize is True


def test_compatible_summary_estimate_matches_plain_json() -> None:
    from agent.context_manager import _msg_budget
    from agent.context_summary import build_summary_request

    snapshot = HistorySnapshot((sm(1, "user", "hello world" * 10),), None, None)
    items = project(snapshot).items
    estimate = _msg_budget(
        items,
        snapshot,
        summary_model=None,
        summary_calibration_factor=1.0,
    )
    # The estimate covers the real request, which carries the configured language rule.
    from agent.config import get_settings

    request_msgs = build_summary_request(
        items, snapshot.checkpoint, language=get_settings().response_language
    )
    assert estimate == len(json.dumps(request_msgs)) // 4


async def test_calibration_factor_fractional_uses_ceil_after_multiply() -> None:
    snapshot = HistorySnapshot((sm(1, "user", "hello world"),), None, None)
    result = await run(
        snapshot,
        mdl=model(window=16000, max_tokens=500),
        calibration_factor=1.37,
    )
    raw = result.raw_tokens
    assert (raw * 1.37) % 1 != 0
    assert result.approximate_tokens == ceil(raw * 1.37)


async def test_no_fit_after_deterministic_preserves_checkpoint_tail_marker() -> None:
    marker = "TAIL_MARKER_XYZ_"
    cp = SummaryCheckpoint(
        text="PREV SUMMARY " + marker,
        covers_until=2,
        source_hash=VALID_HASH,
        version=SUMMARY_VERSION,
    )
    snapshot = HistorySnapshot(big_snapshot().messages, cp, checkpoint_message_id=99)
    with pytest.raises(ContextBudgetError) as exc:
        await run(snapshot, mdl=model(), target_tokens=1750)
    assert exc.value.reason == "no_fit"
    assert exc.value.result.fits is False
    text = "".join(m.get("content") or "" for m in exc.value.result.messages)
    assert marker in text


async def test_stage2_preserves_protected_blocks_exactly() -> None:
    snapshot = stage2_snapshot()
    result = await run(snapshot, mdl=model(), target_tokens=2500)
    assert result.stage == "stage_2"
    msgs = result.messages

    assert msgs[0] == {"role": "system", "content": "You are a helpful agent."}
    assert msgs[1] == {"role": "user", "content": "task one"}
    assert msgs[8] == {"role": "user", "content": "continue please"}

    for index in (9, 11):
        assistant = msgs[index]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "assistant text"
        calls = assistant["tool_calls"]
        assert calls[0]["function"]["name"] == "write_file"
    for index in (10, 12):
        tool = msgs[index]
        assert tool["role"] == "tool"
        assert "Z" * 3000 in tool["content"]


async def test_stage2_reduces_raw_when_stage1_does_not() -> None:
    messages: list[SourceMessage] = [
        sm(1, "system", "sys"),
        sm(2, "user", "task one"),
    ]
    messages += exchange(3, big_chars=3, big_write=2000)
    messages += exchange(5, big_chars=3, big_write=2000)
    messages += exchange(7, big_chars=3, big_write=2000)
    messages.append(sm(9, "user", "continue please"))
    messages += exchange(10, big_chars=3000)
    messages += exchange(12, big_chars=3000)
    snapshot = HistorySnapshot(tuple(messages), None, None)

    none_result = await run(
        snapshot,
        mdl=model(window=64000, max_tokens=1000),
        target_tokens=1000000,
    )
    assert none_result.stage == "none"

    stage2 = await run(snapshot, mdl=model(), target_tokens=2000)
    assert stage2.stage == "stage_2"
    assert stage2.masked_tool_results == 0
    assert stage2.folded_tool_calls == 3
    assert stage2.raw_tokens < none_result.raw_tokens
