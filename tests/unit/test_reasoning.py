from __future__ import annotations

import pytest

from agent.context import ContextManager
from agent.model_config import ReasoningSpec, ResolvedModel
from agent.models import Message
from agent.reasoning import AUTO, decide, validate_effort, wire_parameters


def model(
    efforts: list[str], default: str | None, wire: str | None = "reasoning_effort"
) -> ResolvedModel:
    return ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url="https://llm.example.com/v1",
        api_key="",
        configured=True,
        model="m",
        context_window=32_000,
        max_tokens=4_000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(
            supported=bool(efforts),
            allowed_efforts=efforts,
            default_effort=default,
            wire_parameter=wire,  # type: ignore[arg-type]
        ),
        pricing=None,
    )


FULL = model(["none", "low", "medium", "high", "max"], "medium")
ON_OFF = model(["none", "high"], "high")


def test_message_beats_session_beats_model() -> None:
    assert decide(FULL, message_effort="low", session_effort="high", mode="dev").effective == "low"
    assert decide(FULL, message_effort=None, session_effort="high", mode="dev").effective == "high"
    chosen = decide(FULL, message_effort=None, session_effort=None, mode="dev")
    assert (chosen.effective, chosen.source) == ("medium", "model")


def test_auto_uses_mode_size_and_errors() -> None:
    assert decide(FULL, message_effort=AUTO, session_effort=None, mode="ask").effective == "low"
    assert decide(FULL, message_effort=AUTO, session_effort=None, mode="dev").effective == "medium"
    long_task = decide(FULL, message_effort=AUTO, session_effort=None, mode="dev", task_chars=9000)
    assert long_task.effective == "high"
    one_error = decide(FULL, message_effort=AUTO, session_effort=None, mode="dev", recent_errors=1)
    assert one_error.effective == "high"
    two_errors = decide(FULL, message_effort=AUTO, session_effort=None, mode="ask", recent_errors=2)
    assert two_errors.effective == "max"


def test_auto_picks_the_nearest_allowed_level() -> None:
    chosen = decide(ON_OFF, message_effort=AUTO, session_effort=None, mode="ask")
    assert chosen.effective in ON_OFF.reasoning.allowed_efforts
    assert chosen.source == "auto"


def test_validate_effort() -> None:
    validate_effort(ON_OFF, AUTO)
    validate_effort(ON_OFF, "none")
    with pytest.raises(ValueError, match="does not accept"):
        validate_effort(ON_OFF, "low")


def test_wire_dialects() -> None:
    decision = decide(FULL, message_effort="high", session_effort=None, mode="dev")
    assert wire_parameters(FULL, decision, 4000) == {"reasoning_effort": "high"}
    flag = model(["none", "high"], "high", "enable_thinking")
    off = decide(flag, message_effort="none", session_effort=None, mode="dev")
    assert wire_parameters(flag, off, 4000) == {"enable_thinking": False}
    budget = model(["none", "low", "high"], "high", "thinking_budget")
    on = decide(budget, message_effort="high", session_effort=None, mode="dev")
    assert wire_parameters(budget, on, 4000) == {
        "thinking": {"type": "enabled", "budget_tokens": 3999}
    }


def test_no_reasoning_control_sends_nothing() -> None:
    plain = model([], None, None)
    decision = decide(plain, message_effort=None, session_effort=None, mode="dev")
    assert decision.effective is None
    assert wire_parameters(plain, decision, 4000) == {}


def messages() -> list[Message]:
    return [
        Message(role="system", content="s", kind="normal"),
        Message(role="user", content="first", kind="normal"),
        Message(role="assistant", content="a1", reasoning_content="old thought", kind="normal"),
        Message(role="user", content="second", kind="normal"),
        Message(
            role="assistant",
            content=None,
            reasoning_content="new thought",
            kind="normal",
            tool_calls=[
                {"id": "c", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        ),
        Message(role="tool", content="{}", tool_call_id="c", kind="normal"),
        Message(role="assistant", content="cut", kind="interrupted"),
    ]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("none", []), ("current_turn", ["new thought"]), ("all", ["old thought", "new thought"])],
)
def test_reasoning_history_policy(policy: str, expected: list[str]) -> None:
    prepared = ContextManager(32_000, 4_000, 10_000).prepare(messages(), reasoning_history=policy)
    sent = [item["reasoning_content"] for item in prepared.messages if "reasoning_content" in item]
    assert sent == expected
    assert all(item.get("content") != "cut" for item in prepared.messages)
