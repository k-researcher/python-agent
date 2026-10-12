from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from agent.llm import LLMError, LLMResponse, error_for, stream_error_info, token_count


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


class SSEDecoder:
    """Incremental byte decoder for Server-Sent Events."""

    def __init__(self) -> None:
        self._buffer = b""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        """Consume bytes and return the data values of completed events."""
        self._buffer += chunk
        events: list[str] = []
        while True:
            line, blank = self._take_line()
            if line is None:
                break
            if blank:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data = []
            else:
                self._consume_line(line)
        return events

    def close(self) -> list[str]:
        """Flush the last event when the stream stopped without a blank line."""
        if self._buffer:
            self._consume_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""
        if not self._data:
            return []
        event = "\n".join(self._data)
        self._data = []
        return [event]

    def _take_line(self) -> tuple[bytes | None, bool]:
        buf = self._buffer
        for idx, char in enumerate(buf):
            if char not in (0x0A, 0x0D):
                continue
            if char == 0x0D and idx + 1 == len(buf):
                # A CR at the end of the chunk can be the first half of CRLF; wait for more data.
                return None, False
            line = buf[:idx]
            if char == 0x0D and idx + 1 < len(buf) and buf[idx + 1] == 0x0A:
                self._buffer = buf[idx + 2 :]
            else:
                self._buffer = buf[idx + 1 :]
            return line, line == b""
        return None, False

    def _consume_line(self, line: bytes) -> None:
        if not line or line.startswith(b":"):
            return
        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            self._data.append(value.decode("utf-8", errors="replace"))


class ChatStreamAccumulator:
    """Collect OpenAI chat.completion.chunk payloads into an LLMResponse."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: dict[int, _ToolCallAccumulator] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None

    def add(self, data: str) -> list[StreamEvent]:
        """Process one decoded data value; ``[DONE]`` marks the end of the stream."""
        if data == "[DONE]":
            return [StreamEvent(kind="done")]
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned an invalid stream chunk") from exc
        if not isinstance(payload, dict):
            raise LLMError("LLM returned an invalid stream chunk")
        error = payload.get("error")
        if error:
            info = stream_error_info(error)
            raise error_for(info, f"LLM stream error: {info.message}")

        events: list[StreamEvent] = []
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                self._apply_choice(choice, events)
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._usage = usage
            events.append(StreamEvent(kind="usage"))
        return events

    def _apply_choice(self, choice: dict[str, Any], events: list[StreamEvent]) -> None:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                self._content.append(content)
                events.append(StreamEvent(kind="content", text=content))
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                self._reasoning.append(reasoning)
                events.append(StreamEvent(kind="reasoning", text=reasoning))
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for part in tool_calls:
                    if isinstance(part, dict):
                        self._apply_tool_call(part, events)
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self._finish_reason = finish_reason
            events.append(StreamEvent(kind="finish"))

    def _apply_tool_call(self, part: dict[str, Any], events: list[StreamEvent]) -> None:
        raw_index = part.get("index")
        index = raw_index if isinstance(raw_index, int) else 0
        entry = self._tool_calls.get(index)
        if entry is None:
            entry = _ToolCallAccumulator(index=index)
            self._tool_calls[index] = entry
        tc_id = part.get("id")
        if isinstance(tc_id, str) and tc_id and entry.id is None:
            entry.id = tc_id
        tc_type = part.get("type")
        if isinstance(tc_type, str) and tc_type and entry.type is None:
            entry.type = tc_type
        function = part.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name and entry.name is None:
                entry.name = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry.arguments += arguments
        events.append(StreamEvent(kind="tool_call", index=index))

    def result(self) -> LLMResponse:
        """Build the final LLMResponse from all collected chunks."""
        content = "".join(self._content) or None
        reasoning = "".join(self._reasoning) or None
        tool_calls = [
            {
                "id": entry.id,
                "type": entry.type or "function",
                "function": {
                    "name": entry.name or "",
                    "arguments": entry.arguments,
                },
            }
            for _, entry in sorted(self._tool_calls.items(), key=lambda item: item[0])
        ]
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
