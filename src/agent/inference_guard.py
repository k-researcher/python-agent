"""One cooperative deadline and a shared, bounded cleanup scope per inference."""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from contextvars import ContextVar
from typing import Any, Protocol, TypeVar

from agent.llm_errors import LLMErrorKind, ProviderErrorInfo
from agent.model_config import ResolvedModel

logger = logging.getLogger(__name__)

T = TypeVar("T")
CURRENT_GUARD: ContextVar[GuardContext | None] = ContextVar("inference_guard", default=None)
CURRENT_RUN_GUARD: ContextVar[GuardContext | None] = ContextVar("run_inference_guard", default=None)
_RETAINED: set[asyncio.Future[Any]] = set()


class Clock(Protocol):
    def now(self) -> float: ...
    async def until(self, deadline: float) -> None: ...


class LoopClock:
    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def until(self, deadline: float) -> None:
        await asyncio.sleep(max(0, deadline - self.now()))


def retain[T](task: asyncio.Future[T]) -> asyncio.Future[T]:
    _RETAINED.add(task)

    def done(completed: asyncio.Future[T]) -> None:
        _RETAINED.discard(completed)
        # Task cancellation can retain a traceback with an unfinished database
        # cursor. asyncio.wait() does not retrieve it. Consume it even when the
        # supervisor keeps the completed task, so connection close can finish.
        try:
            completed.exception()
        except asyncio.CancelledError as error:
            traceback.clear_frames(error.__traceback__)

    task.add_done_callback(done)
    return task


async def bounded_cleanup[T](operation: Awaitable[T], clock: Clock, deadline: float) -> T | None:
    task = retain(asyncio.ensure_future(operation))
    timer = retain(asyncio.create_task(clock.until(deadline)))
    try:
        pending: set[asyncio.Future[Any]] = {task, timer}
        done, _ = await asyncio.wait(
            pending, timeout=max(0, deadline - clock.now()), return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        task.cancel()
        return None
    except BaseException as error:
        task.cancel()
        if isinstance(error, Exception):
            logger.warning("Inference cleanup failed: %s", type(error).__name__)
        raise
    finally:
        timer.cancel()


def guard_error(kind: LLMErrorKind, message: str | None = None) -> Exception:
    from agent.llm import LLMError

    explanations = {
        LLMErrorKind.inference_timeout: "Истекло время ожидания ответа. Можно продолжить вручную",
        LLMErrorKind.semantic_idle: "Ответ остановлен: нет новых данных. Можно продолжить вручную",
        LLMErrorKind.repetition: "Обнаружен повтор; ответ остановлен. Можно продолжить вручную",
        LLMErrorKind.empty_response: "Получен пустой ответ",
        LLMErrorKind.invalid_response: "Получен некорректный ответ",
    }
    text = message or explanations.get(kind, kind.value)
    return LLMError(text, info=ProviderErrorInfo(kind, None, kind.value, text, False))


def terminal_reason(reason: BaseException | None) -> str:
    from agent.llm import LLMError
    from agent.session_budget_store import BudgetExceeded

    if isinstance(reason, LLMError):
        return reason.kind.value
    if isinstance(reason, BudgetExceeded):
        return reason.reason
    return "cancelled"


class GuardContext:
    def __init__(
        self,
        model: ResolvedModel,
        *,
        clock: Clock | None = None,
        session_deadline: float | None = None,
    ) -> None:
        self.clock = clock or LoopClock()
        self.t0 = self.clock.now()
        self.deadline = self.t0 + model.inference_timeout_seconds
        self.session_deadline = session_deadline
        self.closed = False
        self.reason: BaseException | None = None
        self.cleanup_deadline: float | None = None
        self.changed = asyncio.Event()
        self.progressed = asyncio.Event()
        self._closers: list[Callable[[], None]] = []
        self.publisher: object | None = None
        self.attempt_usage: tuple[int | None, int | None] = (None, None)

    @property
    def effective_deadline(self) -> float:
        if self.session_deadline is None:
            return self.deadline
        return min(self.deadline, self.session_deadline)

    def observe_usage(self, prompt: int | None, completion: int | None) -> None:
        self.attempt_usage = (
            prompt if prompt is not None else self.attempt_usage[0],
            completion if completion is not None else self.attempt_usage[1],
        )

    def select(self, model: ResolvedModel) -> None:
        narrowed = min(self.deadline, self.t0 + model.inference_timeout_seconds)
        if narrowed < self.deadline:
            self.deadline = narrowed
            self.changed.set()

    def on_close(self, close: Callable[[], None]) -> None:
        if self.closed:
            close()
        else:
            self._closers.append(close)

    def close(self, reason: BaseException | None = None) -> None:
        if self.closed:
            return
        self.closed, self.reason = True, reason
        self.cleanup_deadline = self.clock.now() + 2
        for close in self._closers:
            close()
        self.changed.set()

    def ensure_open(self) -> None:
        if self.closed:
            if self.reason is not None:
                raise self.reason
            raise asyncio.CancelledError

    async def cleanup(self, operation: Awaitable[T]) -> T | None:
        if self.cleanup_deadline is None:
            self.cleanup_deadline = self.clock.now() + 2
        operation_task = retain(asyncio.ensure_future(operation))
        cleanup = retain(
            asyncio.create_task(bounded_cleanup(operation_task, self.clock, self.cleanup_deadline))
        )
        # The observer can cancel the worker again after a guard closes. Cleanup keeps
        # its own deadline and must not be cancelled with that worker.
        return await asyncio.shield(cleanup)

    async def finalize(self, operation: Awaitable[T]) -> T | None:
        task = retain(asyncio.ensure_future(operation))
        if self.closed:
            return await self.cleanup(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.close(asyncio.CancelledError())
            with suppress(BaseException):
                await self.cleanup(task)
            raise

    async def run(self, operation: Coroutine[Any, Any, T], *, cleanup_worker: bool = False) -> T:
        token = CURRENT_GUARD.set(self)
        try:
            worker = retain(asyncio.create_task(operation))
        finally:
            CURRENT_GUARD.reset(token)
        timer: asyncio.Future[None] | None = None
        change: asyncio.Future[bool] | None = None
        try:
            while True:
                self.changed.clear()
                boundary = self.effective_deadline
                timer = retain(asyncio.create_task(self.clock.until(boundary)))
                change = retain(asyncio.create_task(self.changed.wait()))
                waiting: set[asyncio.Future[Any]] = {worker, timer, change}
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                # Model selection and worker completion can happen in the same turn.
                boundary = self.effective_deadline
                if self.closed:
                    self.ensure_open()
                if self.clock.now() >= boundary:
                    if self.session_deadline is not None and boundary == self.session_deadline:
                        from agent.session_budget_store import BudgetExceeded

                        raise BudgetExceeded("deadline_exceeded")
                    raise guard_error(LLMErrorKind.inference_timeout)
                if worker.done():
                    return worker.result()
                timer.cancel()
                change.cancel()
        except BaseException as error:
            self.close(error)
            worker.cancel()
            # A standalone caller can close its loop immediately after this returns.
            # Give its callbacks the same cleanup boundary, without an unbounded join.
            if cleanup_worker:
                with suppress(BaseException):
                    await self.cleanup(asyncio.wait({worker}))
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if change is not None:
                change.cancel()

    async def idle(
        self, operation: Coroutine[Any, Any, T], seconds: float, progress: Callable[[], float]
    ) -> T:
        worker = retain(asyncio.create_task(operation))
        timer: asyncio.Future[None] | None = None
        try:
            while True:
                deadline = progress() + seconds
                timer = retain(asyncio.create_task(self.clock.until(deadline)))
                self.progressed.clear()
                changed = retain(asyncio.create_task(self.progressed.wait()))
                try:
                    waiting: set[asyncio.Future[Any]] = {worker, timer, changed}
                    done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    changed.cancel()
                if self.clock.now() >= progress() + seconds:
                    error = guard_error(LLMErrorKind.semantic_idle)
                    self.close(error)
                    raise error
                if worker in done:
                    return worker.result()
                timer.cancel()
        except BaseException:
            worker.cancel()
            raise
        finally:
            if timer is not None:
                timer.cancel()
