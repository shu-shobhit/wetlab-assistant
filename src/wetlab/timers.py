"""Declared timers, and what happens when two expire at once.

A timer is declared on a step in the corpus and starts when that step commits,
not when it is read: the incubation begins when the scientist has done the thing,
and the pointer moving is the only evidence of that the assistant has.

Expiry is the one event that outranks step speech (architecture section 6), so
the interesting behaviour is not the counting, it is the queue. Two timers that
expire half a second apart both have to be announced, in order, and neither may
be dropped because the other was speaking. That is PS stress case 2.

Two implementations of one interface. `Scheduler` is driven by a clock the caller
advances, which is what the deterministic tests use. `AsyncScheduler` puts the
same logic on `loop.call_later`. Both fire through the same `_fire`, so the tested
path and the live path are the same code.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from .protocol import Timer


@dataclass(frozen=True)
class Pending:
    timer: Timer
    step_id: str
    due_at: float

    def remaining(self, now: float) -> float:
        return max(0.0, self.due_at - now)


class Scheduler:
    """Timers against an injected clock. Nothing fires until `poll` is called."""

    def __init__(
        self,
        on_fire: Callable[[Timer], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_fire = on_fire
        self._clock = clock
        self._pending: list[Pending] = []

    def start(self, timers, step_id: str) -> list[Pending]:
        now = self._clock()
        started = [
            Pending(timer=t, step_id=step_id, due_at=now + t.seconds) for t in timers
        ]
        self._pending.extend(started)
        self._pending.sort(key=lambda p: p.due_at)
        return started

    def cancel(self, timer_id: str) -> bool:
        before = len(self._pending)
        self._pending = [p for p in self._pending if p.timer.id != timer_id]
        return len(self._pending) < before

    def cancel_all(self) -> None:
        self._pending.clear()

    def remaining(self) -> list[tuple[Timer, float]]:
        now = self._clock()
        return [(p.timer, p.remaining(now)) for p in self._pending]

    def pending(self) -> tuple[Pending, ...]:
        """What is scheduled, each with the step it was started for.

        `remaining` drops the step id, which is all a display needs and not
        enough to decide what an advance makes stale. Deciding that from the
        timer id instead means matching a prefix, and step ids are s1 to s17:
        "s12.timer1" starts with "s1".
        """
        return tuple(self._pending)

    def poll(self) -> list[Timer]:
        """Fire everything that is due, earliest first.

        Two timers due in the same poll both fire, in order. Dropping the second
        because the first was mid-announcement is how a wash step gets missed.
        """
        now = self._clock()
        due = [p for p in self._pending if p.due_at <= now]
        if not due:
            return []
        due.sort(key=lambda p: p.due_at)
        self._pending = [p for p in self._pending if p.due_at > now]
        for pending in due:
            self._fire(pending.timer)
        return [p.timer for p in due]

    def _fire(self, timer: Timer) -> None:
        self._on_fire(timer)


class AsyncScheduler(Scheduler):
    """The same scheduler, woken by the event loop instead of by a caller."""

    def __init__(
        self,
        on_fire: Callable[[Timer], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(on_fire, clock=clock)
        self._handles: dict[str, asyncio.TimerHandle] = {}

    def start(self, timers, step_id: str) -> list[Pending]:
        started = super().start(timers, step_id)
        loop = asyncio.get_running_loop()
        for pending in started:
            self._handles[pending.timer.id] = loop.call_later(
                pending.timer.seconds, self._wake, pending.timer.id
            )
        return started

    def _wake(self, timer_id: str) -> None:
        self._handles.pop(timer_id, None)
        self.poll()

    def cancel(self, timer_id: str) -> bool:
        handle = self._handles.pop(timer_id, None)
        if handle is not None:
            handle.cancel()
        return super().cancel(timer_id)

    def cancel_all(self) -> None:
        for handle in self._handles.values():
            handle.cancel()
        self._handles.clear()
        super().cancel_all()
