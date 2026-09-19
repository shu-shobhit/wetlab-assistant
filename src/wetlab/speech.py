"""Everything the assistant says, in the order it says it.

The framework already orders speech: `AgentActivity` keeps a queue of speech
handles and plays them one at a time to completion, and both `session.say()`
and the model's own generated reply join that queue at the same priority. So
this is not a second queue.

What broke version 1 was not a missing queue. It was that a cut was reachable
from ordinary conversation: every repeat emitted one, the hazard guard called
repeat on a false positive, and a cut ran `session.interrupt(force=True)`,
which stopped whatever was playing. The reading cut itself. Nothing here
interrupts, and the only thing that stops speech now is the scientist talking
over it, which the framework handles.

The model is the only voice. A tool changes state and returns words; the model
says them. Anything that has to be said without being asked, a greeting or a
timer expiring, is handed to the model as a fact to pass on rather than spoken
over the top of it. `via_model` is that path.

One utterance is not the model's, and cannot be. A status covers the gap while
a tool runs, and the gap exists precisely because the model is busy producing
the call, so there is nothing to ask. It is written by the model, in the same
response that invokes the tool, and spoken directly. It is cancelled the
instant anything else is about to be heard, so it cannot overlap an answer, and
on the corpus measured it was dropped every time: 12 of 12. That is the one
exception, and it is here rather than hidden.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator

from livekit import rtc

#: How long a status waits before it is spoken. If the tool returns and the
#: real answer is ready first, the status is dropped: "checking the annealing
#: step" followed immediately by the answer is padding, not continuity.
#: Toggled to zero and to a large value in the two evidence runs.
STATUS_DWELL_S = 0.4

class Speech:
    def __init__(self, session, tts, log, sender=None, *, dwell_s: float = STATUS_DWELL_S):
        self._session = session
        self._tts = tts
        self._log = log
        self._sender = sender
        self._dwell = dwell_s
        self._utterances = 0
        self._status_task: asyncio.Task | None = None
        self._status_spoken = False

    # -- context ids --------------------------------------------------------

    def next_context(self) -> str:
        """Claim the next id. The fence, the log and the clips key on this."""
        self._utterances += 1
        return f"u{self._utterances}"

    # -- speaking -----------------------------------------------------------

    async def say(
        self,
        text: str,
        *,
        kind: str,
        step_id: str | None = None,
        allow_interruptions: bool = True,
    ) -> bool:
        """Say something and wait for it to finish. True if it was heard."""
        context_id = self.next_context()
        self._tts.reserve_context(context_id)
        return await self._play(
            text,
            context_id=context_id,
            kind=kind,
            step_id=step_id,
            allow_interruptions=allow_interruptions,
        )

    async def via_model(self, instructions: str, *, kind: str) -> None:
        """Ask the model to say something, rather than saying it directly.

        The model is the only voice. Something that has to be said but does not
        answer a question, a greeting or a timer expiring, is still said by the
        model: it is given the fact and told to pass it on. Two things follow.
        It comes out in the same voice and register as everything else, and it
        cannot arrive on top of an answer, because the framework queues one
        generation behind another.

        A timer is the case where this is a real trade. Saying it directly is
        certain; asking the model to say it is not, because a model can decline
        or reword. The request and what came of it are both logged, so a timer
        that was asked for and never spoken is visible rather than silent.
        """
        # Logged, not sent to the page. The page has no use for it, and the
        # first version sent it anyway: bench_io whitelists what the agent may
        # send, the message was not on the list, and the send raised inside the
        # entrypoint before the greeting or the scripted driver had run. The
        # job crashed on its first utterance. The whitelist was right; sending
        # was the mistake.
        self._log.append("speech.requested", kind=kind, instructions=instructions)

        # Issued, not awaited. The framework queues this behind whatever is
        # already speaking and `_follow_reply` records what came of it. Waiting
        # here would put the caller's progress behind the model's: the caller
        # for the greeting is the entrypoint, and the scripted driver starts on
        # the line after it.
        self._session.generate_reply(instructions=instructions)

    # Pre-synthesis lived here: the next step was rendered by Rime during the
    # pause while the scientist worked, so advancing played audio that already
    # existed. It went with the tools that spoke. The model's words are not
    # known before it writes them, so there is nothing to render in advance,
    # and time to first audio now includes the model composing the step. That
    # cost was accepted when the speech source was made single.
    #
    # The after-step cue, a spoken "Ready.", went the same way. It existed
    # because a finished step, a slow model and a dead worker all sound alike
    # when you cannot see the screen. The model's own utterance ending now
    # marks the end of a step, so a separate cue would be a second voice
    # saying the same thing.

    # -- statuses -----------------------------------------------------------

    def start_status(self, text: str) -> None:
        """Say what is being done, but only if it takes long enough to need it.

        The status arrives in the same model response that invokes the tool, so
        it costs no extra round trip and it is in the words of the question:
        "checking the annealing step", not "let me check".
        """
        if not text or not text.strip():
            return
        self.cancel_status()
        self._status_spoken = False
        self._status_task = asyncio.create_task(self._speak_status(text.strip()))

    async def _speak_status(self, text: str) -> None:
        try:
            await asyncio.sleep(self._dwell)
        except asyncio.CancelledError:
            return
        self._status_spoken = True
        await self.say(text, kind="status")

    def cancel_status(self) -> None:
        """Drop the status if it has not started. Called when a tool returns."""
        task, self._status_task = self._status_task, None
        if task is None or task.done():
            return
        task.cancel()
        if not self._status_spoken:
            self._log.append("status.dropped")

    async def drain_status(self) -> None:
        """Wait for a status already being spoken, so it does not overlap."""
        task = self._status_task
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._status_task = None

    # -- the one place session.say is called --------------------------------

    async def _play(
        self,
        text: str,
        *,
        context_id: str,
        kind: str,
        step_id: str | None,
        allow_interruptions: bool,
        audio: rtc.AudioFrame | None = None,
    ) -> bool:
        # A pending status is dropped the moment anything else is about to be
        # heard. The status exists to cover a gap before the assistant makes
        # any sound, and this is the assistant making a sound, so the gap is
        # over. Without it a speaking tool queued its step and then its own
        # status behind it, and "Moving to the next step" arrived after the
        # step it was announcing. It has to be here rather than in say(),
        # because held audio takes the other path and hits the same problem.
        if kind != "status":
            self.cancel_status()

        self._log.append(
            "speech.start", context_id=context_id, kind=kind, step_id=step_id, text=text
        )
        if self._sender is not None:
            await self._sender.send(
                "speech.start", context_id=context_id, kind=kind, step_id=step_id, text=text
            )

        kwargs = {"allow_interruptions": allow_interruptions, "add_to_chat_ctx": False}
        if audio is not None:
            kwargs["audio"] = _one(audio)
        handle = self._session.say(text, **kwargs)
        await handle.wait_for_playout()
        interrupted = bool(handle.interrupted)

        # An utterance that produced no audio was not heard, whatever the
        # handle says about being interrupted. A stream can end cleanly having
        # spoken nothing: the fence refuses a context cut before it reached the
        # socket, and the framework returns without raising on a 499. Both come
        # back as finished and not interrupted.
        silent = bool(text.strip()) and not self._tts.audio.get(context_id) and audio is None
        played = not interrupted and not silent
        self._log.append(
            "speech.finished",
            context_id=context_id,
            kind=kind,
            step_id=step_id,
            interrupted=interrupted,
            silent=silent,
            played=played,
        )
        if self._sender is not None:
            await self._sender.send("step.state", context_id=context_id, played=played)
        return played


async def _one(frame: rtc.AudioFrame) -> AsyncIterator[rtc.AudioFrame]:
    yield frame
