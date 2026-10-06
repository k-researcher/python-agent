from agent.context import ContextManager
from agent.models import Message


def message(role: str, content: str, index: int) -> Message:
    return Message(id=index, session_id="s", role=role, content=content)


def test_context_preserves_system_first_user_and_recent_messages() -> None:
    source = [message("system", "rules", 1), message("user", "initial task", 2)]
    source.extend(message("assistant", "x" * 400, index) for index in range(3, 30))
    source.append(message("user", "latest question", 30))

    result = ContextManager(2048, 1024, 1000).prepare(source)

    assert result.messages[0]["content"] == "rules"
    assert result.messages[1]["content"] == "initial task"
    assert result.messages[-1]["content"] == "latest question"
    assert result.omitted_messages > 0


def test_context_truncates_large_tool_results() -> None:
    source = [
        message("system", "rules", 1),
        message("user", "task", 2),
        message("tool", "x" * 5000, 3),
    ]

    result = ContextManager(10_000, 1000, 100).prepare(source)

    assert result.truncated_tool_results == 1
    assert "truncated" in result.messages[-1]["content"]
