"""Validate response structure before adaptation or tool execution."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

from agent.inference_guard import guard_error
from agent.llm_errors import LLMErrorKind


def invalid(message: str = "Получен некорректный ответ") -> Exception:
    return guard_error(LLMErrorKind.invalid_response, message)


def envelope(body: Any, *, stream: bool, functions: bool = False) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        raise invalid()
    choices = body.get("choices")
    if not isinstance(choices, list) or (not stream and not choices):
        raise invalid()
    if not choices:
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            raise invalid()
        container = choice.get("delta" if stream else "message", {} if stream else None)
        if not isinstance(container, dict):
            raise invalid()
        for key in ("content", "reasoning", "reasoning_content", "role", "functions_state_id"):
            if container.get(key) is not None and not isinstance(container[key], str):
                raise invalid()
        calls = container.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
                raise invalid()
            for call in calls:
                if stream:
                    index = call.get("index")
                    if type(index) is not int or index < 0:
                        raise invalid()
                for key in ("id", "type", "functions_state_id"):
                    if call.get(key) is not None and not isinstance(call[key], str):
                        raise invalid()
                if call.get("type") not in (None, "function"):
                    raise invalid()
                fn = call.get("function")
                if fn is not None:
                    if not isinstance(fn, dict):
                        raise invalid()
                    for key in ("name", "arguments"):
                        if key in fn and fn[key] is not None and not isinstance(fn[key], str):
                            raise invalid()
        if functions and "function_call" in container:
            fn = container["function_call"]
            if not isinstance(fn, dict):
                raise invalid()
            if "name" in fn and not isinstance(fn["name"], str):
                raise invalid()
            if "arguments" in fn:
                args = fn["arguments"]
                if not isinstance(args, (str, dict)):
                    raise invalid()
            # Completeness belongs after length classification.
    return cast(dict[str, Any], choices[0])


def finish(value: Any, *, functions: bool = False) -> str | None:
    if value is None:
        return None
    if functions and value == "function_call":
        return "tool_calls"
    if not isinstance(value, str) or value not in {
        "stop",
        "tool_calls",
        "length",
        "content_filter",
    }:
        raise invalid("Некорректная причина завершения ответа")
    if value == "content_filter":
        raise invalid("Ответ остановлен фильтром содержимого")
    return value


def arguments_object(value: Any, *, functions: bool = False) -> dict[str, Any]:
    if functions and isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        raise invalid("Отсутствуют аргументы вызова инструмента")
    try:
        result = json.loads(value)
    except ValueError as error:
        raise invalid("Некорректные аргументы вызова инструмента") from error
    if not isinstance(result, dict):
        raise invalid("Аргументы вызова инструмента должны быть объектом")
    return result


def validate_functions(body: Any, *, stream: bool) -> None:
    choice = envelope(body, stream=stream, functions=True)
    if choice is None:
        return
    reason = finish(choice.get("finish_reason"), functions=True)
    container = choice["delta" if stream else "message"]
    if not stream and reason != "length" and "function_call" in container:
        fn = container["function_call"]
        if not isinstance(fn.get("name"), str) or not fn["name"].strip():
            raise invalid()
        arguments_object(fn.get("arguments"), functions=True)


def calls_validated(calls: Any) -> list[dict[str, Any]]:
    if calls is None:
        return []
    if not isinstance(calls, list):
        raise invalid()
    validated = []
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise invalid()
        fn = call.get("function")
        if (
            not isinstance(fn, dict)
            or not isinstance(fn.get("name"), str)
            or not fn["name"].strip()
        ):
            raise invalid()
        arguments_object(fn.get("arguments"))
        call_id = call.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise invalid()
        if not call_id or call_id in seen:
            call_id = f"call_{uuid.uuid4().hex}"
        seen.add(call_id)
        validated.append({**call, "id": call_id})
    return validated


def usable(content: str | None, calls: list[dict[str, Any]], reason: str) -> None:
    if reason == "tool_calls" and not calls:
        raise invalid("Причина tool_calls без вызовов инструментов")
    if not (content or "").strip() and not calls:
        raise guard_error(LLMErrorKind.empty_response)
