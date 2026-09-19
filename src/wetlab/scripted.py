"""Feed a fixed list of turns into a live session, as text.

This replaces harness.py, which ran the agent headless with fixture audio and
scripted pedal presses. That worked when the claims were about a gate, which is
a logic question you can answer without a model or a speech engine. Every claim
now is about the real thing: how long the model and Rime actually take, whether
a spoken status covers the gap, how fast audio stops. A fixture playing computed
silence cannot measure any of it.

So this fakes nothing. The room is real, the model is real, the tools, the
retrieval, Rime and the event log are real. The only thing skipped is the
microphone and the recogniser, and that is the point: the same questions, in the
same order, every run.

Claim 3.3 needs exactly that. It compares the longest silence on tool turns with
statuses on and with them off, and if you ask the questions by talking you say
them differently the second time, the recogniser hears them differently, the
model takes a different path, and the difference you report might be your
wording rather than the feature.

A browser tab still has to join, without speaking, because the silence being
measured is the listener's and that comes from the page's probe.

    WETLAB_SCRIPT=data/scenarios/questions.yaml python -m wetlab.agent dev
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

#: How long to wait after a turn goes quiet before sending the next one. Long
#: enough that a step finishes reading and its cue plays.
SETTLE_S = 1.5

#: Give up on a turn that never goes quiet, so one wedged turn does not hold
#: the whole run open.
TURN_TIMEOUT_S = 90.0


def load(path: Path) -> list[str]:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    return [str(t) for t in (doc.get("turns") or []) if str(t).strip()]


async def play(session, path: Path, log, *, settle_s: float = SETTLE_S) -> None:
    """Say each turn in order, waiting for quiet in between."""
    turns = load(path)
    log.append("script.started", path=str(path), turns=len(turns))
    # The greeting is still being spoken when this starts.
    await _quiet(session, settle_s)

    for number, turn in enumerate(turns, start=1):
        log.append("script.turn", number=number, text=turn)
        try:
            handle = session.generate_reply(user_input=turn)
            await asyncio.wait_for(handle.wait_for_playout(), timeout=TURN_TIMEOUT_S)
            await asyncio.wait_for(_quiet(session, settle_s), timeout=TURN_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.append("script.turn_timeout", number=number, seconds=TURN_TIMEOUT_S)
        except Exception as exc:  # a scripted run should report, not die
            log.append("script.turn_failed", number=number, error=str(exc))

    log.append("script.finished", turns=len(turns))


async def _quiet(session, settle_s: float) -> None:
    """Wait until the assistant has stopped speaking and stayed stopped.

    A tool speaks after the model's reply finishes, so the reply's own handle
    going quiet is not the end of the turn.
    """
    while True:
        await asyncio.sleep(settle_s)
        if session.agent_state != "speaking":
            return
