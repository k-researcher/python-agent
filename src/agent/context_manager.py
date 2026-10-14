"""Context coordinator with progressive compression stages (stage 2.3c).

Starts from a projected history and walks a fixed ladder of stages until a
projection fits the token budget: no-op, tool-result masking, argument
folding, paid summarisation (optional) and finally a deterministic summary
built from whole blocks. The first projection that fits is returned; otherwise
a :class:`ContextBudgetError` carries the best attempt.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

from agent.config import get_settings
from agent.context_blocks import Block, flatten, protected_blocks, split_blocks
from agent.context_budget import Budget, input_budget
from agent.context_compaction import fold_write_arguments, mask_tool_result
from agent.context_projection import (
    HistorySnapshot,
    chain_hash,
    project,
)
from agent.context_summary import (
    SUMMARY_MAX_TOKENS,
    SUMMARY_VERSION,
    SummaryCheckpoint,
    build_summary_request,
    deterministic_summary,
    normalize_summary,
    render_summary_item,
)
from agent.model_config import ResolvedModel
from agent.summary_runner import summary_effort
from agent.wire import WireRequest, build_wire_request, estimate_wire

SummaryRunner = Callable[[list[dict[str, Any]]], Awaitable[str | None]]
CompactionStage = Literal["none", "stage_1", "stage_2", "summary", "deterministic"]


@dataclass(frozen=True, slots=True)
class ContextResult:
    """Outcome of one :meth:`ContextManager.prepare` call."""

    messages: list[dict[str, Any]]
    request: WireRequest
    budget: Budget
    target_tokens: int
    effective_max_tokens: int
    raw_tokens: int
    approximate_tokens: int
    tokens_before: int
    calibration_factor: float
    stage: CompactionStage
    fits: bool
    omitted_messages: int
    masked_tool_results: int
    folded_tool_calls: int
    summary_used: bool
    summary_calls: int
    checkpoint_message_id: int | None
    checkpoint_to_save: SummaryCheckpoint | None
    checkpoint_model: str | None
    would_summarize: bool

    @property
    def truncated_tool_results(self) -> int:
        """Alias kept for compatibility with older callers."""
        return self.masked_tool_results


class ContextBudgetError(RuntimeError):
    """Raised when a projection cannot fit the token budget."""

    reason: str

    def __init__(self, reason: str, result: ContextResult) -> None:
        super().__init__(reason)
        self.reason = reason
        self.result = result


class ContextManager:
    """Coordinate the compression ladder for a single LLM context."""

    def __init__(self, reserved_tokens: int) -> None:
        self.reserved_tokens = reserved_tokens

    async def prepare(
        self,
        snapshot: HistorySnapshot,
        model: ResolvedModel,
        tools: list[dict[str, Any]],
        *,
        calibration_factor: float = 1.0,
        effective_max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_history: str = "none",
        target_tokens: int | None = None,
        summary_runner: SummaryRunner | None = None,
        summary_model_id: str | None = None,
        summary_input_tokens: int | None = None,
        summary_model: ResolvedModel | None = None,
        summary_calibration_factor: float = 1.0,
        stream: bool = True,
    ) -> ContextResult:
        """Build the best fitting projection, walking the stage ladder."""

        def measure(
            items: list[dict[str, Any]],
        ) -> tuple[WireRequest, int, int]:
            request = build_wire_request(
                model,
                items,
                tools,
                reasoning_effort=reasoning_effort,
                max_tokens=effective,
                stream=stream,
            )
            raw = estimate_wire(request)
            approx = ceil(raw * calibration_factor)
            return request, raw, approx

        effective = effective_max_tokens if effective_max_tokens is not None else model.max_tokens
        budget = input_budget(model.context_window, self.reserved_tokens, effective)
        target = (
            budget.input_tokens
            if target_tokens is None
            else min(budget.input_tokens, target_tokens)
        )

        def fits_target(approx: int) -> bool:
            return target > 0 and approx <= target

        proj = project(snapshot, reasoning_history=reasoning_history)
        has_checkpoint = snapshot.checkpoint is not None

        def make_result(
            items: list[dict[str, Any]],
            request: WireRequest,
            raw: int,
            approx: int,
            *,
            stage: CompactionStage,
            masked: int = 0,
            folded: int = 0,
            summary_used: bool = False,
            summary_calls: int = 0,
            checkpoint_to_save: SummaryCheckpoint | None = None,
            checkpoint_model: str | None = None,
            omitted: int = 0,
            would_summarize: bool = False,
            tokens_before: int = 0,
        ) -> ContextResult:
            return ContextResult(
                messages=items,
                request=request,
                budget=budget,
                target_tokens=target,
                effective_max_tokens=effective,
                raw_tokens=raw,
                approximate_tokens=approx,
                tokens_before=tokens_before,
                calibration_factor=calibration_factor,
                stage=stage,
                fits=fits_target(approx),
                omitted_messages=omitted,
                masked_tool_results=masked,
                folded_tool_calls=folded,
                summary_used=summary_used,
                summary_calls=summary_calls,
                checkpoint_message_id=(snapshot.checkpoint_message_id if has_checkpoint else None),
                checkpoint_to_save=checkpoint_to_save,
                checkpoint_model=checkpoint_model,
                would_summarize=would_summarize,
            )

        summary_index = proj.summary_index
        base: list[dict[str, Any]] = []
        base_ids: list[int | None] = []
        for index, item in enumerate(proj.items):
            if index == summary_index:
                continue
            base.append(item)
            base_ids.append(proj.source_ids[index])

        blocks = split_blocks(base)
        protected = protected_blocks(base, blocks, keep_completed=2)

        compressible = frozenset(
            {
                index
                for index, block in enumerate(blocks)
                if index not in protected
                and block.complete
                and block.kind in ("exchange", "assistant", "orphan_tool")
            }
        )

        eligible = frozenset(
            {
                index
                for index, block in enumerate(blocks)
                if index not in protected
                and block.complete
                and block.kind in ("exchange", "assistant", "orphan_tool", "user")
            }
        )

        def assemble(base_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Re-insert the existing checkpoint after the first user item."""
            checkpoint = snapshot.checkpoint
            if checkpoint is None:
                return list(base_items)
            first_user = next(
                (i for i, item in enumerate(base_items) if item.get("role") == "user"),
                None,
            )
            insert_at = first_user + 1 if first_user is not None else len(base_items)
            result = list(base_items)
            result.insert(insert_at, render_summary_item(checkpoint))
            return result

        if budget.input_tokens <= 0 or target <= 0:
            request, raw, approx = measure(proj.items)
            raise ContextBudgetError(
                "invalid_budget",
                make_result(
                    proj.items,
                    request,
                    raw,
                    approx,
                    stage="none",
                    summary_used=has_checkpoint,
                ),
            )

        none_items = assemble(base)
        request_n, raw_n, approx_n = measure(none_items)
        tokens_before = approx_n
        none_result = make_result(
            none_items,
            request_n,
            raw_n,
            approx_n,
            stage="none",
            summary_used=has_checkpoint,
            tokens_before=tokens_before,
        )
        if fits_target(approx_n):
            return none_result

        best_raw = raw_n
        best_items: list[dict[str, Any]] = none_items
        best_request = request_n
        best_approx = approx_n
        best_masked = 0
        best_folded = 0
        best_stage: CompactionStage = "none"

        stage1_raw = _apply_stage1(base, compressible)
        stage1_masked = _count_masked(base, compressible)
        stage1_items = assemble(stage1_raw)
        request_1, raw_1, approx_1 = measure(stage1_items)
        if raw_1 < best_raw:
            best_raw = raw_1
            best_items = stage1_items
            best_request = request_1
            best_approx = approx_1
            best_masked = stage1_masked
            best_stage = "stage_1"
            if fits_target(approx_1):
                return make_result(
                    stage1_items,
                    request_1,
                    raw_1,
                    approx_1,
                    stage="stage_1",
                    masked=stage1_masked,
                    summary_used=has_checkpoint,
                    tokens_before=tokens_before,
                )

        stage2_raw, folded_2 = _apply_stage2(base, compressible)
        stage2_masked = _count_masked(base, compressible)
        stage2_items = assemble(stage2_raw)
        request_2, raw_2, approx_2 = measure(stage2_items)
        if raw_2 < best_raw:
            best_raw = raw_2
            best_items = stage2_items
            best_request = request_2
            best_approx = approx_2
            best_masked = stage2_masked
            best_folded = folded_2
            best_stage = "stage_2"
            if fits_target(approx_2):
                return make_result(
                    stage2_items,
                    request_2,
                    raw_2,
                    approx_2,
                    stage="stage_2",
                    masked=stage2_masked,
                    folded=folded_2,
                    summary_used=has_checkpoint,
                    tokens_before=tokens_before,
                )

        protected_items = assemble(flatten(base, blocks, protected))
        _, _, approx_p = measure(protected_items)
        if not fits_target(approx_p):
            raise ContextBudgetError(
                "protected_over_budget",
                make_result(
                    best_items,
                    best_request,
                    best_raw,
                    best_approx,
                    stage=best_stage,
                    masked=best_masked,
                    folded=best_folded,
                    summary_used=has_checkpoint,
                    tokens_before=tokens_before,
                ),
            )

        first_user_block = next(
            (i for i, block in enumerate(blocks) if block.kind == "user"),
            None,
        )
        prefix: list[int] = []
        if first_user_block is not None:
            for index in range(first_user_block + 1, len(blocks)):
                if index in eligible:
                    prefix.append(index)
                else:
                    break

        while prefix and blocks[prefix[-1]].kind == "user":
            prefix.pop()
        if not prefix:
            raise ContextBudgetError(
                "no_fit",
                make_result(
                    best_items,
                    best_request,
                    best_raw,
                    best_approx,
                    stage=best_stage,
                    masked=best_masked,
                    folded=best_folded,
                    summary_used=has_checkpoint,
                    tokens_before=tokens_before,
                ),
            )

        def prefix_positions(selected: Sequence[int]) -> list[int]:
            positions: list[int] = []
            for block_index in selected:
                block = blocks[block_index]
                positions.extend(range(block.start, block.end))
            return positions

        def checkpoint_for(text: str, selected: Sequence[int]) -> SummaryCheckpoint:
            positions = prefix_positions(selected)
            prefix_ids: list[int] = []
            for position in positions:
                message_id = base_ids[position]
                if message_id is not None:
                    prefix_ids.append(message_id)
            covers = max(prefix_ids) if prefix_ids else 0
            return SummaryCheckpoint(
                text=text,
                covers_until=covers,
                source_hash=chain_hash(
                    snapshot.checkpoint,
                    [{"id": base_ids[p], **base[p]} for p in positions],
                ),
                version=SUMMARY_VERSION,
            )

        def summarised_projection(
            checkpoint: SummaryCheckpoint,
            selected: Sequence[int],
        ) -> tuple[list[dict[str, Any]], int, int]:
            """Build (pre + summary + tail), transforming only compressible blocks.

            Every block after the covered prefix is preserved: protected tail
            exchanges, the last user message and any incomplete exchange are
            kept unchanged; only compressible tail blocks are masked/folded.
            """
            first_prefix = min(selected)
            last_prefix = max(selected)
            pre = flatten(base, blocks, range(first_prefix))
            tail_indices = list(range(last_prefix + 1, len(blocks)))
            tail_compressible = frozenset(i for i in tail_indices if i in compressible)
            tail_transformed, folded = _apply_stage2(base, tail_compressible)
            masked = _count_masked(base, tail_compressible)
            tail = flatten(tail_transformed, blocks, tail_indices)
            return (
                pre + [render_summary_item(checkpoint)] + tail,
                masked,
                folded,
            )

        light_prefix = list(prefix)
        prefix_items = flatten(base, blocks, light_prefix)
        if summary_input_tokens is not None:
            light_prefix, prefix_items = _trim_prefix(
                base,
                blocks,
                light_prefix,
                snapshot,
                summary_input_tokens,
                summary_model=summary_model,
                summary_calibration_factor=summary_calibration_factor,
            )

        summary_calls = 0
        if summary_runner is not None and light_prefix:
            request_msgs = build_summary_request(
                prefix_items, snapshot.checkpoint, language=get_settings().response_language
            )
            summary_calls = 1
            text = await summary_runner(request_msgs)

            if text is not None and text.strip() and _fits_output_budget(text, summary_model):
                norm = normalize_summary(text)
                checkpoint = checkpoint_for(norm, light_prefix)
                s_items, s_masked, s_folded = summarised_projection(checkpoint, light_prefix)
                request_s, raw_s, approx_s = measure(s_items)
                if raw_s < best_raw and fits_target(approx_s):
                    return make_result(
                        s_items,
                        request_s,
                        raw_s,
                        approx_s,
                        stage="summary",
                        masked=s_masked,
                        folded=s_folded,
                        summary_used=True,
                        summary_calls=summary_calls,
                        checkpoint_to_save=checkpoint,
                        checkpoint_model=summary_model_id,
                        omitted=len(prefix_positions(light_prefix)),
                        tokens_before=tokens_before,
                    )

        det_selected = light_prefix
        if det_selected:
            det_prefix_items = flatten(base, blocks, det_selected)
            det_new = deterministic_summary(det_prefix_items, max_chars=SUMMARY_MAX_TOKENS * 4)
            prev_text = snapshot.checkpoint.text if snapshot.checkpoint is not None else None
            det_text = _fit_deterministic_text(prev_text, det_new)
            if det_text is not None:
                det_checkpoint = checkpoint_for(det_text, det_selected)
                det_items, det_masked, det_folded = summarised_projection(
                    det_checkpoint, det_selected
                )
                request_d, raw_d, approx_d = measure(det_items)
                if raw_d < best_raw and fits_target(approx_d):
                    would = summary_runner is None and bool(prefix)
                    return make_result(
                        det_items,
                        request_d,
                        raw_d,
                        approx_d,
                        stage="deterministic",
                        masked=det_masked,
                        folded=det_folded,
                        summary_used=True,
                        summary_calls=summary_calls,
                        checkpoint_to_save=det_checkpoint,
                        checkpoint_model=None,
                        omitted=len(prefix_positions(det_selected)),
                        would_summarize=would,
                        tokens_before=tokens_before,
                    )

        would = summary_runner is None and bool(light_prefix)
        raise ContextBudgetError(
            "no_fit",
            make_result(
                best_items,
                best_request,
                best_raw,
                best_approx,
                stage=best_stage,
                masked=best_masked,
                folded=best_folded,
                summary_used=has_checkpoint,
                tokens_before=tokens_before,
                would_summarize=would,
            ),
        )


def _fits_output_budget(text: str, summary_model: ResolvedModel | None) -> bool:
    """Reject a summary callback response that exceeds the output budget."""
    limit = SUMMARY_MAX_TOKENS
    if summary_model is not None:
        limit = min(SUMMARY_MAX_TOKENS, summary_model.max_tokens)
    return ceil(len(text) / 4) <= limit


def _det_tokens(text: str) -> int:
    """Approximate token count of one piece of summary text."""
    return ceil(len(text) / 4)


def _join_det(previous: str | None, new: str) -> str:
    """Join the previous checkpoint (in full) with the new deterministic facts."""
    if previous is None:
        return new
    return "Earlier summary:\n" + previous + "\n\n" + new


def _fit_deterministic_text(previous: str | None, new: str) -> str | None:
    """Return a checkpoint text within SUMMARY_MAX_TOKENS, or None to decline.

    The previous checkpoint is always retained in full. Only NEW records are
    shortened, by dropping whole record lines; the previous checkpoint is
    never sliced. When the previous checkpoint alone leaves no room for any
    new record, None is returned so the caller declines instead of cutting
    earlier compressed data.
    """
    if _det_tokens(_join_det(previous, new)) <= SUMMARY_MAX_TOKENS:
        return _join_det(previous, new)
    lines = new.splitlines()
    while lines:
        candidate = _join_det(previous, "\n".join(lines))
        if _det_tokens(candidate) <= SUMMARY_MAX_TOKENS:
            return candidate
        lines.pop()
    return None


def _msg_budget(
    items: list[dict[str, Any]],
    snapshot: HistorySnapshot,
    *,
    summary_model: ResolvedModel | None,
    summary_calibration_factor: float,
) -> int:
    """Approximate token budget of the summary request under the chosen rules."""
    request_msgs = build_summary_request(
        items, snapshot.checkpoint, language=get_settings().response_language
    )
    if summary_model is not None:
        request = build_wire_request(
            summary_model,
            request_msgs,
            [],
            reasoning_effort=summary_effort(summary_model),
            max_tokens=min(SUMMARY_MAX_TOKENS, summary_model.max_tokens),
            stream=False,
        )
        return ceil(estimate_wire(request) * summary_calibration_factor)
    return _json_chars(request_msgs) // 4


def _trim_prefix(
    base: list[dict[str, Any]],
    blocks: list[Block],
    prefix: Sequence[int],
    snapshot: HistorySnapshot,
    summary_input_tokens: int,
    *,
    summary_model: ResolvedModel | None,
    summary_calibration_factor: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Drop trailing whole blocks until the summary request fits its budget."""
    selected = list(prefix)
    prefix_items = flatten(base, blocks, selected)
    while (
        len(selected) > 1
        and _msg_budget(
            prefix_items,
            snapshot,
            summary_model=summary_model,
            summary_calibration_factor=summary_calibration_factor,
        )
        > summary_input_tokens
    ):
        selected = selected[:-1]

        while selected and blocks[selected[-1]].kind == "user":
            selected.pop()
        if not selected:
            return [], []
        prefix_items = flatten(base, blocks, selected)
    if (
        len(selected) == 1
        and _msg_budget(
            prefix_items,
            snapshot,
            summary_model=summary_model,
            summary_calibration_factor=summary_calibration_factor,
        )
        > summary_input_tokens
    ):
        return [], []
    return selected, prefix_items


def _json_chars(items: list[dict[str, Any]]) -> int:
    return len(json.dumps(items))


def _masked_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Mask a tool item only when the result does not grow."""
    if item.get("role") != "tool":
        return None
    new = mask_tool_result(item)
    if len(json.dumps(new)) <= len(json.dumps(item)):
        return new
    return None


def _apply_stage1(
    items: list[dict[str, Any]], compressible: frozenset[int]
) -> list[dict[str, Any]]:
    result = [dict(item) for item in items]
    blocks = split_blocks(items)
    for block_index, block in enumerate(blocks):
        if block_index not in compressible:
            continue
        for position in range(block.start, block.end):
            new = _masked_item(items[position])
            if new is not None:
                result[position] = new
    return result


def _count_masked(items: list[dict[str, Any]], compressible: frozenset[int]) -> int:
    blocks = split_blocks(items)
    count = 0
    for block_index, block in enumerate(blocks):
        if block_index not in compressible:
            continue
        for position in range(block.start, block.end):
            if _masked_item(items[position]) is not None:
                count += 1
    return count


def _apply_stage2(
    items: list[dict[str, Any]], compressible: frozenset[int]
) -> tuple[list[dict[str, Any]], int]:
    result = [dict(item) for item in items]
    blocks = split_blocks(items)
    folded = 0
    for block_index, block in enumerate(blocks):
        if block_index not in compressible:
            continue
        for position in range(block.start, block.end):
            item = items[position]
            new = _masked_item(item)
            if new is not None:
                result[position] = new
                continue
            if item.get("role") == "assistant" and item.get("tool_calls"):
                folded_item = fold_write_arguments(item)
                if (
                    len(json.dumps(folded_item)) <= len(json.dumps(item))
                    and folded_item["tool_calls"] != item["tool_calls"]
                ):
                    result[position] = folded_item
                    folded += 1
    return result, folded
