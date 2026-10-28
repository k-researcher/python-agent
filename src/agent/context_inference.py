from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import ceil
from typing import Any

from agent.attempt_audit import finish_attempt, start_attempt
from agent.calibration_store import Calibration, load_calibration
from agent.config import Settings
from agent.context_budget import input_budget, overflow_target
from agent.context_manager import ContextBudgetError, ContextManager, ContextResult
from agent.context_state import (
    HistoryChanged,
    LoadedHistory,
    ensure_active,
    load_history,
    persist_checkpoint,
)
from agent.context_summary import SUMMARY_MAX_TOKENS, normalize_summary
from agent.database import session_factory
from agent.events import StreamPublisher, emit_event
from agent.inference_guard import CURRENT_GUARD, GuardContext, retain, terminal_reason
from agent.llm import AttemptRecord, LLMClient, LLMError, LLMResponse
from agent.llm_errors import TERMINAL_GUARDS, LLMErrorKind, is_context_overflow
from agent.model_config import ResolvedModel, ResolvedModelRegistry
from agent.models import Session as SessionRow
from agent.reasoning import ReasoningDecision
from agent.session_budget_store import BudgetExceeded, check_attempt_budget
from agent.summary_runner import summary_effort
from agent.token_calibration import CalibrationKey
from agent.usage_ledger import attempt_detail
from agent.wire import (
    WireRequest,
    body_sha256,
    build_wire_request,
    endpoint_of,
    estimate_wire,
    wire_version,
)

logger = logging.getLogger(__name__)

MAX_HTTP_ATTEMPTS = 12
MAX_SUMMARY_CALLS = 4
MAX_CONTEXT_RETRIES = 2
TRANSIENT_KINDS = frozenset(
    {
        LLMErrorKind.rate_limit,
        LLMErrorKind.server,
        LLMErrorKind.timeout,
        LLMErrorKind.network,
        LLMErrorKind.stream_incomplete,
    }
)


@dataclass(slots=True)
class InferenceState:
    inference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    http_attempts: int = 0
    summary_calls: int = 0
    overflow_retries: int = 0
    started: bool = False
    current_model_id: str | None = None
    visited_models: set[str] = field(default_factory=set)
    sent_requests: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class PreparedContext:
    context: ContextResult
    history: LoadedHistory
    calibration: Calibration


class ContextPreparationError(ContextBudgetError):
    def __init__(self, error: ContextBudgetError, prepared: PreparedContext) -> None:
        super().__init__(error.reason, error.result)
        self.prepared = prepared


@dataclass(frozen=True, slots=True)
class InferenceResult:
    response: LLMResponse
    model: ResolvedModel
    reasoning: ReasoningDecision
    prepared: PreparedContext
    publisher: StreamPublisher | None = None


def context_stats(prepared: PreparedContext, model: ResolvedModel) -> dict[str, Any]:
    result = prepared.context
    checkpoint = result.checkpoint_to_save or prepared.history.snapshot.checkpoint
    return {
        "model_id": model.id,
        "provider": model.provider_id,
        "wire_version": result.request.wire_version,
        "raw_tokens": result.raw_tokens,
        "tokens_before": result.tokens_before,
        "approximate_tokens": result.approximate_tokens,
        "calibration_factor": result.calibration_factor,
        "calibration_samples": prepared.calibration.samples,
        "context_window": result.budget.context_window,
        "reserved_tokens": result.budget.context_window - result.budget.input_tokens,
        "input_budget": result.budget.input_tokens,
        "output_tokens": result.budget.output_tokens,
        "effective_max_tokens": result.effective_max_tokens,
        "safety_margin": result.budget.safety_margin,
        "target_tokens": result.target_tokens,
        "stage": result.stage,
        "fits": result.fits,
        "omitted_messages": result.omitted_messages + prepared.history.covered_messages,
        "truncated_tool_results": result.truncated_tool_results,
        "masked_tool_results": result.masked_tool_results,
        "folded_tool_calls": result.folded_tool_calls,
        "summary_used": result.summary_used,
        "summary_calls": result.summary_calls,
        "summary_checkpoint_id": result.checkpoint_message_id,
        "covers_until": checkpoint.covers_until if checkpoint else None,
        "would_summarize": result.would_summarize,
        "error": None,
    }


class ContextInference:
    def __init__(
        self,
        session_id: str,
        reserved_tokens: int,
        definitions: list[dict[str, Any]],
        models: ResolvedModelRegistry,
        client_for: Callable[[ResolvedModel], LLMClient],
        reasoning_for: Callable[[ResolvedModel], ReasoningDecision],
        settings: Settings | None = None,
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.manager = ContextManager(reserved_tokens)
        self.reserved_tokens = reserved_tokens
        self.definitions = definitions
        self.models = models
        self.client_for = client_for
        self.reasoning_for = reasoning_for
        self.state = InferenceState()
        self.publisher: StreamPublisher | None = None

    async def event(self, event_type: str, payload: dict[str, Any]) -> None:
        guard = CURRENT_GUARD.get()
        if guard is not None:
            guard.ensure_open()
        async with session_factory() as db:
            await emit_event(db, event_type, payload, self.session_id)

    async def _send(
        self,
        model: ResolvedModel,
        client: LLMClient,
        request: WireRequest,
        calibration: Calibration,
        purpose: str,
        publisher: StreamPublisher | None = None,
    ) -> LLMResponse:
        audit_id: int | None = None
        guard = CURRENT_GUARD.get()
        if guard is not None:
            guard.ensure_open()
            guard.select(model)
        initial: AttemptRecord | None = None
        key = CalibrationKey(model.provider_id, model.model, request.wire_version)

        async def attempt(record: AttemptRecord) -> None:
            nonlocal audit_id, initial
            if record.status == "started":
                if guard is not None:
                    guard.ensure_open()
                    guard.attempt_usage = (None, None)
                initial = record
                await ensure_active(self.session_id)
                if self.settings is not None:
                    async with session_factory() as db:
                        session = await db.get(SessionRow, self.session_id)
                        if session is not None:
                            await check_attempt_budget(
                                db,
                                self.settings,
                                session,
                                request_bytes=len(request.body),
                                max_output_tokens=int(request.payload["max_tokens"]),
                                pricing=model.pricing,
                                now=datetime.now(UTC),
                            )
                if self.state.http_attempts >= MAX_HTTP_ATTEMPTS:
                    raise LLMError("HTTP attempt limit exhausted")
                if purpose == "summary":
                    if self.state.summary_calls >= MAX_SUMMARY_CALLS:
                        raise LLMError("Summary attempt limit exhausted")
                    self.state.summary_calls += 1
                self.state.http_attempts += 1
                recording = retain(
                    asyncio.create_task(
                        start_attempt(
                            self.session_id,
                            {
                                "destination": endpoint_of(model),
                                "operation": f"chat:{model.id}",
                                "inference_id": self.state.inference_id,
                                "purpose": purpose,
                                "provider": model.provider_id,
                                "model_id": model.id,
                                "model": model.model,
                                "wire_version": request.wire_version,
                                "raw_estimate": estimate_wire(request),
                                "factor": calibration.factor,
                                "detail": attempt_detail(int(request.payload["max_tokens"])),
                            },
                            record,
                            pricing=model.pricing,
                        )
                    )
                )
                try:
                    audit_id = await asyncio.shield(recording)
                except asyncio.CancelledError:
                    # The row may appear late; finish that same attempt when its ID arrives.
                    async def late_terminal() -> None:
                        late_id = await asyncio.shield(recording)
                        reason = guard.reason if guard is not None else None
                        kind = terminal_reason(reason)
                        terminal = finish_attempt(
                            late_id,
                            replace(
                                record,
                                status="error",
                                error_kind=kind,
                                prompt_tokens=guard.attempt_usage[0] if guard else None,
                                completion_tokens=guard.attempt_usage[1] if guard else None,
                            ),
                            key,
                        )
                        if guard is not None:
                            await guard.cleanup(terminal)
                        else:
                            await terminal

                    def recorded(completed: asyncio.Future[int]) -> None:
                        if not completed.cancelled() and completed.exception() is None:
                            retain(asyncio.create_task(late_terminal()))

                    recording.add_done_callback(recorded)
                    if guard is not None:
                        with suppress(BaseException):
                            await guard.cleanup(recording)
                    raise
                if guard is not None:
                    guard.ensure_open()
                await ensure_active(self.session_id)
            elif audit_id is not None:
                if initial is not None:
                    record = replace(
                        record,
                        prompt_tokens=record.prompt_tokens
                        if record.prompt_tokens is not None
                        else initial.prompt_tokens,
                        completion_tokens=record.completion_tokens
                        if record.completion_tokens is not None
                        else initial.completion_tokens,
                    )
                initial = record
                if guard is not None and guard.closed:
                    reason = guard.reason
                    kind = terminal_reason(reason)
                    record = replace(record, status="error", error_kind=kind)
                    await guard.cleanup(finish_attempt(audit_id, record, key))
                else:
                    await finish_attempt(audit_id, record, key)

        async def delta(kind: str, text: str) -> None:
            if guard is not None:
                guard.ensure_open()
            if text:
                self.state.started = True
                if publisher is not None:
                    await publisher.add(kind, text)

        if hasattr(client, "send"):
            try:
                response = await client.send(
                    request,
                    on_attempt=attempt,
                    on_delta=delta if publisher else None,
                )
                if guard is not None:
                    guard.ensure_open()
                return validated_response(response)
            except BaseException as error:
                if guard is not None and (
                    isinstance(error, asyncio.CancelledError)
                    or isinstance(error, LLMError)
                    and error.kind in TERMINAL_GUARDS
                ):
                    guard.close(error)
                if (
                    audit_id is not None
                    and initial is not None
                    and guard is not None
                    and guard.closed
                ):
                    reason = guard.reason
                    kind = terminal_reason(reason)
                    # Cleanup never replaces the first terminal reason.
                    with suppress(BaseException):
                        await guard.cleanup(
                            finish_attempt(
                                audit_id,
                                replace(
                                    initial,
                                    status="error",
                                    error_kind=kind,
                                    prompt_tokens=initial.prompt_tokens
                                    if initial.prompt_tokens is not None
                                    else guard.attempt_usage[0],
                                    completion_tokens=initial.completion_tokens
                                    if initial.completion_tokens is not None
                                    else guard.attempt_usage[1],
                                ),
                                key,
                            )
                        )
                raise
        initial = AttemptRecord(0, "started", len(request.body), body_sha256(request))
        await attempt(initial)
        try:
            response = await client.chat(
                request.payload["messages"],
                request.payload.get("tools", []),
                reasoning_effort=request.payload.get("reasoning_effort"),
                max_tokens=request.payload["max_tokens"],
                on_delta=delta if publisher else None,
            )
            if guard is not None:
                guard.ensure_open()
            response = validated_response(response)
        except BaseException as error:
            if isinstance(error, BudgetExceeded):
                raise
            if guard is not None and (
                isinstance(error, asyncio.CancelledError)
                or isinstance(error, LLMError)
                and error.kind in TERMINAL_GUARDS
            ):
                guard.close(error)
            await attempt(
                replace(
                    initial,
                    status="error",
                    error_kind=(
                        "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else error.kind.value
                        if isinstance(error, LLMError)
                        else "unknown"
                    ),
                )
            )
            raise
        await attempt(
            replace(
                initial,
                status="completed",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        )
        return response

    async def prepare(
        self,
        model: ResolvedModel,
        decision: ReasoningDecision,
        *,
        effective_max_tokens: int | None = None,
        target_tokens: int | None = None,
        paid_summary: bool = True,
    ) -> PreparedContext:
        light = self.models.light()
        async with session_factory() as db:
            history = await load_history(db, self.session_id)
            calibration = await load_calibration(
                db,
                CalibrationKey(
                    model.provider_id,
                    model.model,
                    wire_version(model),
                ),
            )
            summary_calibration = await load_calibration(
                db,
                CalibrationKey(light.provider_id, light.model, wire_version(light)),
            )
        summary_budget = input_budget(
            light.context_window,
            0,
            min(SUMMARY_MAX_TOKENS, light.max_tokens),
        )

        async def summarize(items: list[dict[str, Any]]) -> str | None:
            if (
                self.state.summary_calls >= MAX_SUMMARY_CALLS
                or self.state.http_attempts >= MAX_HTTP_ATTEMPTS
            ):
                return None
            dedicated = replace(light, max_retries=0)
            request = build_wire_request(
                dedicated,
                items,
                [],
                reasoning_effort=summary_effort(dedicated),
                max_tokens=min(SUMMARY_MAX_TOKENS, light.max_tokens),
                stream=False,
            )
            if (
                ceil(estimate_wire(request) * summary_calibration.factor)
                > summary_budget.input_tokens
            ):
                await self.event(
                    "context.summary_failed",
                    {"kind": "budget", "summary_calls": self.state.summary_calls},
                )
                return None
            client = LLMClient(dedicated)
            try:
                response = await self._send(
                    dedicated, client, request, summary_calibration, "summary"
                )
            except LLMError as error:
                if error.kind in TERMINAL_GUARDS:
                    guard = CURRENT_GUARD.get()
                    if guard is not None:
                        guard.close(error)
                    raise
                await self.event(
                    "context.summary_failed",
                    {"kind": error.kind.value, "summary_calls": self.state.summary_calls},
                )
                return None
            finally:
                guard = CURRENT_GUARD.get()
                if guard is not None and guard.closed:
                    with suppress(BaseException):
                        await guard.cleanup(client.aclose())
                else:
                    if guard is not None:
                        await guard.finalize(client.aclose())
                    else:
                        await client.aclose()
            raw_text = response.content or ""
            text = normalize_summary(raw_text) if raw_text.strip() else ""
            if (
                not text
                or response.tool_calls
                or response.finish_reason == "length"
                or ceil(len(text) / 4) > min(SUMMARY_MAX_TOKENS, light.max_tokens)
                or (
                    response.completion_tokens is not None
                    and response.completion_tokens > min(SUMMARY_MAX_TOKENS, light.max_tokens)
                )
            ):
                await self.event(
                    "context.summary_failed",
                    {"kind": "invalid_response", "summary_calls": self.state.summary_calls},
                )
                return None
            return text

        can_summarize = (
            paid_summary and light.configured and self.state.summary_calls < MAX_SUMMARY_CALLS
        )
        try:
            context = await self.manager.prepare(
                history.snapshot,
                model,
                self.definitions,
                calibration_factor=calibration.factor,
                effective_max_tokens=effective_max_tokens,
                reasoning_effort=decision.effective,
                reasoning_history=model.reasoning.history,
                target_tokens=target_tokens,
                summary_runner=summarize if can_summarize else None,
                summary_model_id=light.id,
                summary_input_tokens=summary_budget.input_tokens,
                summary_model=light,
                summary_calibration_factor=summary_calibration.factor,
                stream=True,
            )
        except ContextBudgetError as error:
            error = ContextBudgetError(
                error.reason, replace(error.result, summary_calls=self.state.summary_calls)
            )
            raise ContextPreparationError(
                error,
                PreparedContext(error.result, history, calibration),
            ) from error
        return PreparedContext(
            replace(context, summary_calls=self.state.summary_calls), history, calibration
        )

    async def run(self, primary: ResolvedModel) -> InferenceResult:
        guard = CURRENT_GUARD.get()
        if guard is None:
            return await GuardContext(primary).run(self.run(primary), cleanup_worker=True)
        guard.ensure_open()
        guard.select(primary)
        try:
            return await self._run(primary)
        except ContextBudgetError as error:
            await self.event(
                "context.error",
                {
                    "reason": error.reason,
                    "input_budget": error.result.budget.input_tokens,
                    "target_tokens": error.result.target_tokens,
                    "model_id": self.state.current_model_id or primary.id,
                },
            )
            raise

    async def _run(self, primary: ResolvedModel) -> InferenceResult:
        model_ids = [primary.id, *self.models.fallback]
        last_error: LLMError | None = None
        last_model_id = primary.id
        for model_id in model_ids:
            if model_id in self.state.visited_models:
                continue
            self.state.visited_models.add(model_id)
            model = self.models.models.get(model_id)
            if model is None or (model_id != primary.id and not model.configured):
                continue
            guard = CURRENT_GUARD.get()
            if guard is not None:
                guard.ensure_open()
                guard.select(model)
            self.state.current_model_id = model.id
            if last_error is not None:
                await self.event(
                    "llm.fallback",
                    {"from": last_model_id, "to": model.id, "reason": last_error.kind.value},
                )
            decision = self.reasoning_for(model)
            target: int | None = None
            effective = model.max_tokens
            output_retries = 0
            source_retries = 0
            allow_paid = True
            previous_raw: int | None = None
            while True:
                try:
                    if guard is not None:
                        guard.ensure_open()
                    prepared = await self.prepare(
                        model,
                        decision,
                        effective_max_tokens=effective,
                        target_tokens=target,
                        paid_summary=allow_paid,
                    )
                    if guard is not None:
                        guard.ensure_open()
                    context = prepared.context
                    if previous_raw is not None and context.raw_tokens >= previous_raw:
                        raise ContextBudgetError("no_progress", context)
                    if context.checkpoint_to_save is not None:
                        checkpoint_id = await persist_checkpoint(
                            self.session_id,
                            prepared.history,
                            context.checkpoint_to_save,
                            context.checkpoint_model,
                        )
                        if guard is not None:
                            guard.ensure_open()
                        context = replace(context, checkpoint_message_id=checkpoint_id)
                        prepared = replace(prepared, context=context)
                    fingerprint = body_sha256(context.request)
                    if fingerprint in self.state.sent_requests:
                        raise ContextBudgetError("no_progress", context)
                    self.state.sent_requests.add(fingerprint)
                    if context.stage != "none" or context.summary_used:
                        payload = context_stats(prepared, model)
                        payload.update(
                            inference_id=self.state.inference_id,
                            reason="overflow"
                            if target is not None
                            else "fallback"
                            if last_error
                            else "budget",
                            overflow_retry=self.state.overflow_retries,
                        )
                        await self.event("context.compacted", payload)
                    await ensure_active(self.session_id)
                    publisher = self.publisher = StreamPublisher(self.session_id)
                    if guard is not None:
                        guard.publisher = publisher
                        guard.on_close(publisher.close)
                    try:
                        response = await self._send(
                            model,
                            self.client_for(model),
                            context.request,
                            prepared.calibration,
                            "inference",
                            publisher,
                        )
                        if guard is not None:
                            guard.ensure_open()
                    except BaseException:
                        publisher.close()
                        if guard is not None and guard.closed:
                            await guard.cleanup(publisher.reset())
                        else:
                            await publisher.reset()
                        raise
                    else:
                        if response.finish_reason == "length":
                            publisher.close()
                            await publisher.reset()
                        else:
                            await publisher.flush()
                            publisher.close()
                    if (
                        response.content
                        or response.reasoning
                        or response.tool_calls
                        or response.interrupted_tool_calls
                    ):
                        self.state.started = True
                    if (
                        response.finish_reason == "length"
                        and not self.state.started
                        and output_retries < 2
                    ):
                        ceiling = model.context_window // 4
                        increased = min(ceiling, max(effective + 1, effective * 3 // 2))
                        if increased > effective:
                            effective = increased
                            output_retries += 1
                            continue
                    return InferenceResult(response, model, decision, prepared, publisher)
                except HistoryChanged:
                    source_retries += 1
                    allow_paid = False
                    if source_retries > 2:
                        raise
                except LLMError as error:
                    self.state.started = self.state.started or error.started
                    if self.state.started:
                        raise
                    if is_context_overflow(error.info):
                        if self.state.overflow_retries >= MAX_CONTEXT_RETRIES:
                            raise ContextBudgetError("retry_exhausted", context) from error
                        previous_raw = context.raw_tokens
                        target = min(
                            overflow_target(context.budget),
                            int(context.raw_tokens * context.calibration_factor * 0.8),
                            context.target_tokens,
                        )
                        self.state.overflow_retries += 1
                        await self.event(
                            "context.overflow",
                            {
                                "model_id": model.id,
                                "code": error.info.code,
                                "target": target,
                                "overflow_retry": self.state.overflow_retries,
                            },
                        )
                        continue
                    if (
                        error.kind not in TRANSIENT_KINDS
                        or self.state.http_attempts >= MAX_HTTP_ATTEMPTS
                    ):
                        raise
                    last_error = error
                    last_model_id = model.id
                    break
        if last_error is not None:
            raise last_error
        raise LLMError("No available inference model")


def validated_response(response: LLMResponse) -> LLMResponse:
    """Apply the same acceptance rules to substituted clients."""
    from agent.llm import LLMClient as ResponseClient

    result = ResponseClient._parse_response(
        {
            "choices": [
                {
                    "finish_reason": response.finish_reason,
                    "message": {
                        "content": response.content,
                        "reasoning_content": response.reasoning,
                        "tool_calls": response.tool_calls,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        }
    )
    if response.interrupted_tool_calls is not None:
        result.interrupted_tool_calls = response.interrupted_tool_calls
    return result
