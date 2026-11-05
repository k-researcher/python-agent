from __future__ import annotations

import pytest

from agent.context import ContextManager
from agent.model_config import ReasoningSpec, ResolvedModel
from agent.models import Message, Session
from agent.reasoning import AUTO, decide, validate_effort, wire_parameters
from agent.runtime import AgentSupervisor


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


LEVELS = ["none", "minimal", "low", "medium", "high", "max"]
MEASURED = model(["none", "minimal", "low", "high", "max"], "minimal")


@pytest.mark.parametrize(
    "plain", [model(["none", "low", "medium", "high", "max"], None), model(LEVELS, "none")]
)
def test_auto_without_a_model_default_uses_mode_size_and_errors(plain: ResolvedModel) -> None:
    def auto(mode: str, task_chars: int = 0, recent_errors: int = 0) -> str | None:
        chosen = decide(
            plain,
            message_effort=AUTO,
            session_effort=None,
            mode=mode,
            task_chars=task_chars,
            recent_errors=recent_errors,
        )
        return chosen.effective

    assert auto("ask") == "low"
    assert auto("dev") == "medium"
    assert auto("dev", task_chars=9000) == "high"
    assert auto("dev", recent_errors=1) == "high"
    assert auto("ask", recent_errors=2) == "max"


def test_auto_starts_from_the_model_default_and_raises_it_step_by_step() -> None:
    def auto(mode: str, task_chars: int = 0, recent_errors: int = 0) -> tuple[str | None, str]:
        chosen = decide(
            MEASURED,
            message_effort=AUTO,
            session_effort=None,
            mode=mode,
            task_chars=task_chars,
            recent_errors=recent_errors,
        )
        return chosen.effective, chosen.reason

    assert auto("ask") == ("minimal", "model default")
    assert auto("dev") == ("minimal", "model default")
    assert auto("dev", task_chars=9000) == ("low", "long task")
    assert auto("dev", recent_errors=1) == ("low", "one failed step")
    assert auto("dev", task_chars=9000, recent_errors=1) == ("high", "one failed step")
    assert auto("ask", recent_errors=2) == ("max", "two failed steps in a row")


def test_auto_picks_the_nearest_allowed_level() -> None:
    chosen = decide(ON_OFF, message_effort=AUTO, session_effort=None, mode="ask")
    assert chosen.effective in ON_OFF.reasoning.allowed_efforts
    assert chosen.source == "auto"


def test_auto_does_not_go_back_down_within_a_turn() -> None:
    kept = decide(FULL, message_effort=AUTO, session_effort=None, mode="dev", turn_effort="high")
    assert (kept.effective, kept.source) == ("high", "auto")
    raised = decide(
        FULL,
        message_effort=AUTO,
        session_effort=None,
        mode="dev",
        recent_errors=2,
        turn_effort="high",
    )
    assert raised.effective == "max"
    lower = decide(FULL, message_effort=AUTO, session_effort=None, mode="dev", turn_effort="low")
    assert lower.effective == "medium"
    unknown = decide(
        FULL, message_effort=AUTO, session_effort=None, mode="dev", turn_effort="xhigh"
    )
    assert unknown.effective == "medium"
    fixed = decide(FULL, message_effort="low", session_effort=None, mode="dev", turn_effort="max")
    assert fixed.effective == "low"


def turn_message(role: str, effort: str | None = None, **fields: object) -> Message:
    return Message(role=role, kind="normal", skipped=False, reasoning_effort=effort, **fields)


def test_a_successful_step_keeps_the_raised_auto_level_until_the_next_message() -> None:
    call = [{"id": "c", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
    session = Session(mode="dev", configuration={})
    session.messages = [
        turn_message("user", AUTO, content="first task"),
        turn_message("assistant", "max", content="done"),
        turn_message("user", AUTO, content="second task"),
        turn_message("assistant", "medium", tool_calls=call),
        turn_message("tool", content='{"success": false}', tool_call_id="c"),
        turn_message("assistant", "high", tool_calls=call),
        turn_message("tool", content='{"success": true}', tool_call_id="c"),
    ]
    assert AgentSupervisor._reasoning_for(session, FULL).effective == "high"

    session.messages = [*session.messages, turn_message("user", AUTO, content="third task")]
    assert AgentSupervisor._reasoning_for(session, FULL).effective == "medium"


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


def test_lower_effort_steps_down_to_the_next_allowed_level() -> None:
    from agent.reasoning import lower_effort

    target = model(["none", "low", "high"], "low")
    assert lower_effort(target, "high") == "low"
    assert lower_effort(target, "low") == "none"
    assert lower_effort(target, "none") is None
    # A level the model does not list still steps down to an allowed one.
    assert lower_effort(target, "medium") == "low"
    assert lower_effort(target, None) is None
    assert lower_effort(target, "unknown") is None
