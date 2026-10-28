from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from agent.llm import LLMResponse, error_for, stream_error_info, token_count
from agent.llm_errors import LLMErrorKind, ProviderErrorInfo
from agent.reasoning_tags import Part, ReasoningTagSplitter
from agent.response_validation import calls_validated, envelope, finish, invalid, usable

if TYPE_CHECKING:
    from agent.model_config import ReasoningTagsSpec


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["content", "reasoning", "tool_call", "usage", "finish", "done"]
    text: str = ""
    index: int | None = None


@dataclass
class _ToolCallAccumulator:
    index: int
    id: str | None = None
    type: str | None = None
    name: str | None = None
    arguments: str = ""
    # Provider state of the call in the functions dialect; sent back with the history.
    state_id: str | None = None


class SSEDecoder:
    """Incremental bounded SSE decoder; iter_feed parses only the requested event."""

    MAX_BYTES = 256 * 1024
    MAX_LINES = 4096

    def __init__(self) -> None:
        self._line = bytearray()
        self._data: list[str] = []
        self._bytes = 0
        self._lines = 0
        self._cr = False
        self._cr_counted = False

    def iter_feed(self, chunk: bytes) -> Iterator[str]:
        """Alias for incremental consumers."""
        return self.feed(chunk)

    def feed(self, chunk: bytes) -> Iterator[str]:
        for char in chunk:
            if self._cr:
                self._cr = False
                if char == 10:
                    if self._cr_counted:
                        self._count(1)
                    continue
            if char in (10, 13):
                self._cr = char == 13
                self._cr_counted = bool(self._line)
                if not self._line:
                    if self._data:
                        value = "\n".join(self._data)
                        self._data = []
                        self._bytes = self._lines = 0
                        yield value
                    else:
                        self._bytes = self._lines = 0
                    continue
                self._count(1)
                self._lines += 1
                if self._lines > self.MAX_LINES:
                    raise invalid("Слишком много строк в событии потока")
                self._consume_line(bytes(self._line))
                self._line.clear()
            else:
                self._count(1)
                self._line.append(char)

    def _count(self, count: int) -> None:
        if self._bytes + count > self.MAX_BYTES:
            raise invalid("Слишком большое событие потока")
        self._bytes += count

    def close(self) -> list[str]:
        if self._line:
            self._lines += 1
            if self._lines > self.MAX_LINES:
                raise invalid()
            self._consume_line(bytes(self._line))
            self._line.clear()
        if not self._data:
            return []
        event = "\n".join(self._data)
        self._data = []
        self._bytes = self._lines = 0
        return [event]

    def _consume_line(self, line: bytes) -> None:
        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            self._data.append(value.decode("utf-8", errors="replace"))


class ChatStreamAccumulator:
    """Collect OpenAI chat.completion.chunk payloads into an LLMResponse."""

    def __init__(
        self, reasoning_tags: ReasoningTagsSpec | None = None, *, defer_text: bool = False
    ) -> None:
        self._defer_text = defer_text
        self.done = False
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: dict[int, _ToolCallAccumulator] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        # With tags, a reasoning block inside the content goes to the reasoning.
        self._splitter = (
            ReasoningTagSplitter(reasoning_tags.start, reasoning_tags.end)
            if reasoning_tags is not None
            else None
        )

    def add(self, data: str) -> list[StreamEvent]:
        """Process one decoded data value; ``[DONE]`` marks the end of the stream."""
        if self.done:
            return []
        if data == "[DONE]":
            self.done = True
            events: list[StreamEvent] = []
            if self._splitter is not None:
                self._add_parts(self._splitter.close(), events)
            events.append(StreamEvent(kind="done"))
            return events
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise invalid("LLM returned an invalid stream chunk") from exc
        return self.add_payload(payload)

    def add_payload(self, payload: Any) -> list[StreamEvent]:
        """Consume one already decoded event, without parsing another JSON copy."""
        if self.done:
            return []
        if not isinstance(payload, dict):
            raise invalid("LLM returned an invalid stream chunk")
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._usage = usage
        error = payload.get("error")
        if error:
            info = stream_error_info(error)
            raise error_for(info, f"LLM stream error: {info.message}")

        envelope(payload, stream=True)
        events: list[StreamEvent] = []
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                self._apply_choice(choice, events)
        if isinstance(usage, dict):
            events.append(StreamEvent(kind="usage"))
        return events

    def _apply_choice(self, choice: dict[str, Any], events: list[StreamEvent]) -> None:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                if self._splitter is not None:
                    self._add_parts(self._splitter.feed(content), events)
                else:
                    if not self._defer_text:
                        self._content.append(content)
                    events.append(StreamEvent(kind="content", text=content))
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                if not self._defer_text:
                    self._reasoning.append(reasoning)
                events.append(StreamEvent(kind="reasoning", text=reasoning))
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for part in tool_calls:
                    if isinstance(part, dict):
                        self._apply_tool_call(part, events)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            reason = finish(finish_reason)
            if self._finish_reason is not None and self._finish_reason != reason:
                raise invalid("Конфликтующие причины завершения ответа")
            self._finish_reason = reason
            events.append(StreamEvent(kind="finish"))

    def _add_parts(self, parts: list[Part], events: list[StreamEvent] | None) -> None:
        """Store split parts; add stream events when ``events`` is given."""
        for kind, text in parts:
            if not self._defer_text or events is None:
                self.accept_text(kind, text)
            if events is not None:
                events.append(StreamEvent(kind=kind, text=text))

    def accept_text(self, kind: str, text: str) -> None:
        """Store text after its caller has checked the channel guard."""
        (self._content if kind == "content" else self._reasoning).append(text)

    def end_events(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self._splitter is not None:
            self._add_parts(self._splitter.close(), events)
        return events

    def _apply_tool_call(self, part: dict[str, Any], events: list[StreamEvent]) -> None:
        raw_index = part.get("index")
        index = raw_index if isinstance(raw_index, int) else 0
        entry = self._tool_calls.get(index)
        state_id = part.get("functions_state_id")
        if set(part) <= {"index", "functions_state_id"}:
            # A late state ID belongs to a call that already started; it never opens one.
            if entry is not None and isinstance(state_id, str) and state_id:
                if entry.state_id is not None and entry.state_id != state_id:
                    raise invalid("Конфликтующее состояние вызова")
                entry.state_id = state_id
            return
        function_part = part.get("function") or {}
        if entry is None and not (
            part.get("id") or function_part.get("name") or function_part.get("arguments")
        ):
            return
        if entry is None:
            entry = _ToolCallAccumulator(index=index)
            self._tool_calls[index] = entry
        tc_id = part.get("id")
        if isinstance(tc_id, str) and tc_id:
            if entry.id is not None and entry.id != tc_id:
                raise invalid("Конфликтующие идентификаторы вызова")
            entry.id = tc_id
        tc_type = part.get("type")
        if isinstance(tc_type, str) and tc_type and entry.type is None:
            entry.type = tc_type
        function = part.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                if entry.name is not None and entry.name != name:
                    raise invalid("Конфликтующие имена вызова")
                entry.name = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry.arguments += arguments
        if isinstance(state_id, str) and state_id:
            if entry.state_id is not None and entry.state_id != state_id:
                raise invalid("Конфликтующее состояние вызова")
            entry.state_id = state_id
        events.append(StreamEvent(kind="tool_call", index=index))

    @property
    def usage_counts(self) -> tuple[int | None, int | None]:
        return token_count(self._usage, "prompt_tokens"), token_count(
            self._usage, "completion_tokens"
        )

    def finalize(self) -> LLMResponse:
        """Accept a framed, usable response; result() remains a raw state snapshot."""
        result = self.result()
        if not self.done and self._finish_reason is None:
            message = "LLM stream ended without a completion boundary"
            raise error_for(
                ProviderErrorInfo(LLMErrorKind.stream_incomplete, None, None, message, True),
                message,
                started=bool(result.content or result.reasoning or result.tool_calls),
            )
        if result.finish_reason == "length":
            result.interrupted_tool_calls = result.tool_calls or None
            result.tool_calls = []
            return result
        if any(entry.type != "function" for entry in self._tool_calls.values()):
            raise invalid("Отсутствует тип вызова инструмента")
        result.tool_calls = calls_validated(result.tool_calls)
        usable(result.content, result.tool_calls, result.finish_reason)
        return result

    def result(self) -> LLMResponse:
        """Snapshot accumulated state, without accepting it as a completed response."""
        if self._splitter is not None:
            # A stream without "[DONE]" can still hold a tail.
            self._add_parts(self._splitter.close(), None)
        content = "".join(self._content) or None
        reasoning = "".join(self._reasoning) or None
        tool_calls: list[dict[str, Any]] = []
        for _, entry in sorted(self._tool_calls.items(), key=lambda item: item[0]):
            call: dict[str, Any] = {
                "id": entry.id,
                "type": entry.type or "function",
                "function": {"name": entry.name or "", "arguments": entry.arguments},
            }
            if entry.state_id is not None:
                call["functions_state_id"] = entry.state_id
            tool_calls.append(call)
        usage = self._usage or {}
        prompt_tokens = token_count(usage, "prompt_tokens")
        completion_tokens = token_count(usage, "completion_tokens")
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._finish_reason or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning=reasoning,
        )
