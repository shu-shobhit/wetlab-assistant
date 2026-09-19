"""The worker: one LiveKit session, one protocol, one conversation.

Everything is done by talking. There is no pedal, no key and no command
vocabulary to learn. The scientist says what they want, the model picks a tool,
and the tool moves a pointer or reads a step.

Four processes, unchanged: `livekit-server --dev` routes audio and data on
loopback, this worker holds every key and all the state, an aiohttp server
mints join tokens and serves the page, and the browser tab is a microphone, a
speaker and a screen that decides nothing.

What this file is not any more is a state machine and an adapter for it. The
version it replaces emitted command objects that a Runtime executed, so that a
gate could be tested as a pure function. Nothing gates, so the indirection went
with the gate.

Run, after `livekit-server --dev`:

    scripts/bench_up.sh
    open http://127.0.0.1:8080
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import os
import re
from pathlib import Path

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from . import bench_io, events, markup, position as position_mod, protocol as protocol_mod
from . import retrieval as retrieval_mod
from . import settings as settings_mod
from . import speech as speech_mod
from . import timers as timers_mod
from . import tools as tools_mod
from . import tts_rime

#: How the page names a room so the agent knows what to read: the separator
#: between the prefix, the protocol id and the random part. A dot, because a
#: protocol id is lowercase words joined by underscores and never holds one.
ROOM_SEP = "."

settings_mod.load_dotenv(override=False)


#: The rendering rules, written out concretely rather than as a general
#: instruction to "speak naturally". They are the same rules the offline
#: preparation pass follows, and they are here because the two mechanisms cover
#: different things: the standing context gives the model the prepared spoken
#: form of a step, which covers values it repeats, and these rules cover
#: sentences it composes itself.
#:
#: The markup, punctuation and sentence-length rules come from Rime's own
#: guidance for prompting a model whose output it will speak: there is no
#: expressiveness API, so the prompt is the only lever on delivery, and a model
#: emits SSML unprompted which Rime then reads out bracket by bracket.
#:
#: One piece of that guidance is deliberately not here. Rime suggests seeding
#: light disfluencies, "um" and "well", to make a conversational agent sound
#: human. This one is read while somebody pipettes, and a hesitation in front
#: of a volume is noise in the one place noise is expensive.
SPEAKING_RULES = """\
Everything you say is spoken aloud by a speech engine, never read on a screen.
Write what should be heard, using these rules, which are the same ones the
protocol text was prepared with:

- Write every number as words. "twenty five", not "25". "zero point five", not
  "0.5". Say a decimal digit by digit after the point.
- Write every unit out in full, as it is said: "microlitres", not "µl";
  "degrees celsius", not "°C"; "millimolar", not "mM".
- Write plain words and nothing else. Do not write spell(), and do not space
  letters apart to make them be read out. Acronyms that a person reads letter
  by letter are handled after you, so writing DNA is correct and enough.
- Write anything that is not said the way it is written as it is said instead.
      MgCl2 -> "magnesium chloride", M0492 -> "M zero four nine two"
- Never write a digit. If you find yourself about to, write the word instead.
- No markup, ever. No SSML, no <break>, no asterisks, no bullet points, no
  headings. spell() is the only thing you may write that is not plain speech.
  Anything else is read out literally, including the brackets.
- Punctuation is the only control you have over how it sounds. A comma is a
  short pause, a full stop is a longer one, a question mark lifts the end of
  the sentence. Put a comma before a value the listener has to act on.
- Keep a sentence under twenty five words, and under fifteen where you can.
"""

SYSTEM_PROMPT = f"""\
You are a voice assistant for a scientist working at a laboratory bench. Their
hands are busy and gloved and they cannot look at a screen, so everything
happens by talking.

You read a written protocol aloud and answer questions about it. You keep a
pointer to where they are. You do not check that they have done anything, and
you never claim they have.

How to behave:

- Be brief. One or two sentences. This is speech, and a paragraph is unbearable
  to listen to when your hands are busy.
- When they ask you to read, move, or start something, call the tool. It
  returns the step in the wording to say it in. Say that wording, word for
  word, and change nothing in it. It has been prepared so that every number
  and unit is said correctly, and a value you rephrase is a value that can be
  wrong. You may add a few words of your own before or after it.
- Say each step once. The tool that moved you already gave you the words, so
  there is nothing to look up afterwards and nothing to repeat.
- When they ask something you cannot answer from the steps in front of you,
  call look_up with their own words as the query.
- Answer only from this protocol. If it is not in the protocol, say so plainly
  rather than answering from general knowledge. A plausible invented volume is
  worse than "the protocol does not say".
- Every tool takes a `status`: one short sentence saying what you are about to
  do, in the words of the question you were asked. "Checking the annealing
  step", not "let me check". It is spoken only if the tool takes long enough to
  need it.
- When they say they are done with the protocol, call finish_protocol. Saying
  it has ended without calling it leaves the timers running, and a timer that
  goes off at an empty bench is a warning nobody hears.
- If they interrupt you, answer what they asked, then ask whether to carry on.
  Do not silently resume, and do not start again from the top.

{SPEAKING_RULES}"""


def setup(proc) -> None:
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.55)


server = AgentServer(setup_fnc=setup)


def build_tts(s) -> tts_rime.FencedRimeTTS:
    kwargs = dict(
        model="mistv3",
        lang="eng",
        sample_rate=22050,
        pause_between_brackets=True,
        phonemize_between_brackets=True,
        api_key=s.rime_api_key,
    )
    if s.rime_speaker:
        kwargs["speaker"] = s.rime_speaker
    return tts_rime.FencedRimeTTS(**kwargs)


#: OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def build_llm(s):
    """The conversation model, over OpenRouter.

    Built here rather than through the plugin's `with_openrouter`, which
    assembles the OpenRouter body itself out of fallbacks, provider preferences
    and plugins and offers no way to add to it. `reasoning` has to go in that
    body, and everything else the helper does is the base URL and reading the
    key from the environment, both of which are supplied here.

    Reasoning is worth a setting because the model is where the time goes: on a
    live run the first tool call arrived 2.6 to 4.9 seconds after the turn was
    committed, and a tool turn pays that twice.
    """
    extra: dict = {}
    if s.llm_reasoning == "off":
        extra["reasoning"] = {"enabled": False}
    elif s.llm_reasoning:
        # An effort level rather than a switch, because not every model has a
        # switch. `z-ai/glm-5.3-flash` answers 400 to `enabled: false` saying
        # reasoning is mandatory for the endpoint, and answers `minimal` with
        # zero reasoning tokens.
        extra["reasoning"] = {"effort": s.llm_reasoning}
    if s.llm_provider:
        # Tried in the order given, and never outside it.
        #
        # Left to route, OpenRouter balances price against speed, and on these
        # models the cheap hosts are the slow ones: the same
        # `deepseek/deepseek-v4-flash` answers in 0.84 s on one provider and
        # 6.43 s on another. A run that quietly lands on the second has timings
        # that describe a different machine from the one in its header.
        #
        # `allow_fallbacks: false` keeps it inside the list, and the list is
        # what makes that safe. A single pin turns one provider's 429 into a
        # failed turn; naming two means a busy one degrades to the next fast one
        # instead, and still never to whatever happens to be cheapest.
        order = [p.strip() for p in s.llm_provider.split(",") if p.strip()]
        extra["provider"] = {"order": order, "allow_fallbacks": False}
    return lk_openai.LLM(
        model=s.llm_model,
        api_key=s.openrouter_api_key,
        base_url=OPENROUTER_URL,
        temperature=0.2,
        # One tool at a time. Two pointer moves in one turn is not something
        # this product has an answer for.
        parallel_tool_calls=False,
        extra_body=extra,
    )


def _run_dir(s) -> Path:
    """One directory per run, named by when it started.

    A label can be given with WETLAB_RUN_LABEL. Run directories are ignored by
    git and there is nothing else to tell a scratch run from the one behind a
    published number, so the demo run and an evidence run say so in their own
    names.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", os.environ.get("WETLAB_RUN_LABEL", "")).strip("-")
    return s.runs_dir / (f"{stamp}-{label}" if label else stamp)


def protocol_for_room(room_name: str, corpus_dir: Path, default: str) -> tuple[str, str]:
    """Which protocol this room asks for, and why that one.

    The room name is the only channel the choice can travel on. A LiveKit agent
    job is dispatched when the room is *created*, so by the time there is a
    session for the page to send a message to, the protocol has already had to
    be loaded. The page therefore puts the id into the name it asks for a token
    for: `bench.<protocol_id>.<random>`.

    The name comes from the browser, so it is checked against what is actually
    in the corpus rather than trusted, and anything else falls back to the
    default. An unknown id that was honoured would put the agent on a different
    protocol from the one named on screen, which is the one failure here that
    would not be obvious while it was happening.

    Returns the id and a short reason, which goes in the run header so a log
    says not just which protocol was read but why that one.
    """
    parts = str(room_name or "").split(ROOM_SEP)
    if len(parts) < 3 or not parts[1].strip():
        return default, "room named no protocol; used the default"
    asked = parts[1].strip()
    if asked not in {p.id for p in protocol_mod.available(corpus_dir)}:
        return default, f"room asked for {asked!r}, which is not in the corpus; used the default"
    return asked, "chosen on the page"


class Assistant:
    """The state of one conversation, and the actions the tools call.

    Everything the model can cause runs through here, so there is one place to
    read to know what the assistant can do.
    """

    def __init__(self, *, protocol, speech, log, sender=None, scheduler=None):
        self.protocol = protocol
        self.position = position_mod.Position(protocol)
        self.index = retrieval_mod.from_protocol(protocol)
        self.speech = speech
        self.log = log
        self.sender = sender
        self._scheduler = scheduler or timers_mod.Scheduler(self.on_timer)
        self._ad_hoc = 0
        #: Ids of timers the scientist asked for. No advance may cancel one.
        self._asked: set[str] = set()

    def attach_scheduler(self, scheduler) -> None:
        """Swap in the real scheduler once it can be given this as a callback."""
        self._scheduler = scheduler

    # -- what the tools call ------------------------------------------------

    @contextlib.asynccontextmanager
    async def doing(self, status: str):
        """Wrap a tool call: offer the status, then take it back.

        The status is spoken only if the tool outlasts the dwell. A lookup that
        returns at once would otherwise produce "checking the annealing step"
        followed immediately by the answer, which is padding rather than
        continuity.
        """
        self.speech.start_status(status)
        try:
            yield
        finally:
            self.speech.cancel_status()
            await self.speech.drain_status()

    async def present_step(self) -> str:
        """Hand the current step to the model to say, and record that it was.

        This used to speak. A tool called it, it played Rime audio, and the
        framework then spoke the model's reply about the same turn: two
        utterances from two sources for one thing the scientist asked for, and
        the scientist heard every step twice. A tool doing output as a side
        effect of moving a pointer is the cause, so the tool no longer does
        output. It moves the pointer and returns the words.

        The step is marked read here rather than when audio finishes. That is
        weaker than it was: it records that the step was given to the model to
        say, not that anybody heard it. The gate that used to depend on the
        difference is gone, and the one thing that still reads this ledger is
        `look_up`, which says whether a step has come up yet.
        """
        step = self.position.current
        text = self.protocol.readable_for(step)
        if self.sender is not None:
            await self.sender.send(
                "step.begin",
                step_id=step.id,
                number=step.number,
                total=self.position.total,
                # The page had nothing to show without this. It painted the step
                # from the `speech.start` that a speaking tool emitted, and that
                # went when tools stopped speaking, so from then on the panel
                # rendered a heading over an empty body. Sending it here ties
                # what is on screen to the same call that hands the step to the
                # model, so the two cannot drift apart.
                #
                # The readable form, markup stripped. Only Rime sees spell().
                text=text,
                # And the step as the protocol writes it. The page shows both,
                # because the difference between them is the offline preparation
                # pass: "25 µl" on one line and "twenty five microlitres" on the
                # next is the whole of claim 3.1, visible without a log.
                source=step.text,
            )
        self.position.mark_read(step.id)
        # Reading a step is what carrying on looks like, so it undoes an ending
        # without needing to be asked whether that is what was meant.
        self.position.ended = False
        self.log.append("step.presented", step_id=step.id, number=step.number)
        return text

    def advance(self) -> None:
        """Move on, start what the finished step declared, stop what is stale.

        The timer starts here rather than when the reading ends, and the
        difference is the point. An incubation is measured from when the tube
        went in, not from when the sentence about it stopped, and those differ
        by however long the step took to carry out.

        Any timer still running from an earlier step is cancelled. Saying you
        are done with a step means you are done with what came before it.

        Stale is decided on the step a timer was started for, not on a prefix of
        its id. Step ids run s1 to s17, so "s12.timer1" starts with "s1", and a
        timer from step twelve survived an advance out of step one.

        A timer the scientist asked for is never stale. It belongs to no step,
        and its id belongs to no step either, so the prefix test made every one
        of them stale on the next advance: "set a timer for ten minutes"
        followed by "next step" lost it, and said nothing.
        """
        finished = self.position.current
        stale = [
            p
            for p in self._scheduler.pending()
            if p.step_id != finished.id and p.timer.id not in self._asked
        ]
        for pending in stale:
            self._cancel_timer(pending.timer.id, "advanced")
        self.position.next()
        if finished.timers:
            self._start_timers(finished.timers, finished.id)

    def _start_timers(self, timers, step_id: str) -> None:
        self._scheduler.start(timers, step_id)
        declared = [{"id": t.id, "label": t.label, "seconds": t.seconds} for t in timers]
        self.log.append("timers.started", step_id=step_id, timers=declared)
        if self.sender is not None:
            asyncio.create_task(
                self.sender.send("timers.started", step_id=step_id, timers=declared)
            )

    def start_ad_hoc_timer(self, seconds: float, label: str) -> None:
        self._ad_hoc += 1
        timer = protocol_mod.Timer(id=f"ad_hoc.{self._ad_hoc}", label=label, seconds=seconds)
        self._asked.add(timer.id)
        self._start_timers((timer,), self.position.current.id)

    def _cancel_timer(self, timer_id: str, reason: str) -> bool:
        """Stop one timer, and tell the log and the page that it stopped.

        The page runs its own countdown off `timers.started` and has no other
        way to learn that one ended early. Cancelling was logged and not sent,
        so a timer dropped by an advance kept counting down on screen for the
        rest of the session, and the display and the scheduler disagreed with
        only the log knowing which was right.
        """
        if not self._scheduler.cancel(timer_id):
            return False
        self._asked.discard(timer_id)
        self.log.append("timer.cancelled", timer_id=timer_id, reason=reason)
        if self.sender is not None:
            asyncio.create_task(
                self.sender.send("timer.cancelled", timer_id=timer_id, reason=reason)
            )
        return True

    def finish(self) -> str:
        """End the protocol: stop everything running, and record that it ended.

        There was no way to do this. Asked to end the protocol, the model said
        "okay, ending the protocol here at step twelve", called nothing, and the
        session carried on with a twenty minute incubation still counting down
        on screen. Claiming a thing was done when nothing was done is the one
        behaviour the instructions are written to prevent, and the fix for it is
        a tool rather than another sentence telling it not to.

        The timers are the reason this is not cosmetic. A protocol ends when
        somebody walks away from the bench, and a timer that fires at an empty
        bench is an announcement nobody hears about a step nobody is doing.
        """
        stopped = [pending.timer.id for pending in self._scheduler.pending()]
        for timer_id in stopped:
            self._cancel_timer(timer_id, "finished")
        self.position.ended = True
        self.log.append(
            "protocol.finished",
            step_id=self.position.current.id,
            number=self.position.number,
            timers_stopped=len(stopped),
        )
        if self.sender is not None:
            asyncio.create_task(
                self.sender.send(
                    "protocol.finished",
                    number=self.position.number,
                    total=self.position.total,
                    timers_stopped=len(stopped),
                )
            )
        where = f"ended at step {self.position.number} of {self.position.total}"
        if not stopped:
            return f"{where}; no timers were running"
        return f"{where}; stopped {len(stopped)} running timer" + ("s" if len(stopped) > 1 else "")

    def cancel_timer(self, label: str) -> bool:
        for timer, _ in self.remaining_timers():
            if label.lower() in timer.label.lower() or label == timer.id:
                return self._cancel_timer(timer.id, "asked")
        return False

    def remaining_timers(self):
        return self._scheduler.remaining()

    def spoken_text(self, step_id: str) -> str:
        """A step's text as the model should see it, markup removed."""
        step = next((s for s in self.protocol.steps if s.id == step_id), None)
        return self.protocol.readable_for(step) if step is not None else ""

    def where(self, *, with_text: bool = False) -> str:
        """Where the pointer is, for the model."""
        pos = self.position
        out = f"step {pos.number} of {pos.total}"
        if with_text:
            out += f": {self.protocol.readable_for(pos.current)}"
        return out

    # -- standing context ---------------------------------------------------

    def context_block(self) -> str:
        """Assembled fresh for each model call.

        The window is the prepared spoken form rather than the source text, so
        a value the model repeats is already in the wording that gets it said
        correctly. The contents are the whole protocol, one line per step, so
        it can answer "which step was the one about the primers" without a
        lookup.
        """
        pos = self.position
        state = ""
        if pos.ended:
            state = " (ended; they said they were done with it, and the timers were stopped)"
        elif not pos.started:
            state = " (not started; they have not asked you to begin)"
        lines = [
            f"Protocol: {self.protocol.title}",
            f"Position: step {pos.number} of {pos.total}{state}",
        ]
        running = self.remaining_timers()
        if running:
            lines.append(
                "Timers running: "
                + "; ".join(f"{t.label}, {s:.0f}s left" for t, s in running)
            )
        lines.append("")
        lines.append("The current step and the ones before it, as they were spoken:")
        for step in pos.window():
            marker = ">" if step.index == pos.index else " "
            lines.append(f"{marker} {step.number}. {self.protocol.readable_for(step)}")
        lines.append("")
        lines.append("Every step in this protocol, by number:")
        for number, opening in pos.contents():
            lines.append(f"  {number}. {opening}")
        return "\n".join(lines)

    # -- timers firing ------------------------------------------------------

    def on_timer(self, timer) -> None:
        # A timer that has gone off is no longer one an advance could cancel.
        self._asked.discard(timer.id)
        self.log.append("timer.fired", timer_id=timer.id, label=timer.label)
        asyncio.create_task(self._announce(timer))

    async def _announce(self, timer) -> None:
        if self.sender is not None:
            await self.sender.send("timer.fired", timer_id=timer.id, label=timer.label)
        await self.speech.via_model(
            f"The timer for {timer.label} has just finished. Tell them so, in one "
            "short sentence, and say nothing else. Do not read a step and do not "
            "ask a question.",
            kind="announce",
        )


class WetlabAgent(Agent):
    def __init__(self, app: Assistant) -> None:
        super().__init__(instructions=SYSTEM_PROMPT, tools=tools_mod.build_tools(app))
        self._app = app
        #: Decided by the preparation pass, per protocol, and read out of what
        #: it wrote. Nothing here decides which terms are spelled.
        self._spelled = markup.spelled_terms(
            s.speech for s in app.protocol.spoken.values()
        )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Tell the model where it is, in the words the step is spoken in."""
        turn_ctx.add_message(role="system", content=self._app.context_block())

    def tts_node(self, text, model_settings):
        """The one place text becomes audio, and so the one place markup goes.

        `spell(DNA)` is an instruction to Rime rather than a word. It used to
        be carried in the corpus text, which meant the model was shown it, and
        the model repeated it: Rime was handed the word "spell" to say and the
        listener heard it.

        Applying it here instead means the model never sees markup and cannot
        leak it, and that a term the model wrote in a sentence of its own gets
        spelled too, which under the old arrangement it did not.
        """
        return Agent.default.tts_node(
            self, markup.stream(text, self._spelled), model_settings
        )


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    s = settings_mod.load()
    protocol_id, why = protocol_for_room(ctx.room.name, s.corpus_dir, s.protocol_id)
    proto = protocol_mod.load(s.corpus_dir, protocol_id)

    log = events.Log(
        _run_dir(s),
        {
            "protocol": proto.id,
            "protocol_dir": protocol_id,
            "protocol_chosen": why,
            "citation": proto.citation,
            "room": ctx.room.name,
            "model": "mistv3",
            "speaker": s.rime_speaker or "(default)",
            "llm": s.llm_model,
            # Which of the two runs a figure came from is not recoverable from
            # the model name alone: the same model with reasoning on and off is
            # two different response times.
            "llm_reasoning": s.llm_reasoning,
            # The host, not just the model. The same model is served at six
            # times the latency by one provider as by another, so a timing
            # without this names only half of what produced it.
            "llm_provider": s.llm_provider or "(openrouter routed)",
            "stt": s.stt_model,
            "steps": len(proto.steps),
            "prepared": len(proto.spoken),
            # Which code produced this run. Every published figure comes from a
            # log, so a log that cannot be tied to a commit cannot be checked
            # against the code that made it.
            "revision": events.revision(),
            "label": os.environ.get("WETLAB_RUN_LABEL", ""),
            "script": os.environ.get("WETLAB_SCRIPT", ""),
        },
    )

    await ctx.connect()
    vad = ctx.proc.userdata["vad"]
    tts = build_tts(s)
    session = AgentSession(
        # Streaming, through the LiveKit inference gateway. The recogniser this
        # replaces was a chat completion over a whole buffered utterance, so a
        # transcript arrived three to five seconds after the speech ended and
        # every user turn cut the reading long after the fact.
        stt=inference.STT(model=s.stt_model),
        llm=build_llm(s),
        tts=tts,
        vad=vad,
        use_tts_aligned_transcript=True,
        turn_handling={
            "turn_detection": inference.TurnDetector(),
            # A recogniser that emits while you speak can afford to wait less
            # before deciding you have stopped.
            "endpointing": {"min_delay": 0.3, "max_delay": 2.5},
            "interruption": {
                # The gateway classifies a backchannel against the audio itself.
                # The old two-word floor was approximating that with a counter,
                # so it goes rather than fighting it.
                "mode": "adaptive",
                "min_duration": 0.5,
                "min_words": 0,
            },
            # Off, because in this agent it cannot ever pay.
            #
            # It starts the model on an interim transcript and keeps the result
            # if nothing about the turn changed by the time it is confirmed.
            # `on_user_turn_completed` adds the context block to every turn, so
            # something always changes: a live run logged "preemptive generation
            # invalidated" on twelve of thirteen turns. It bought no latency and
            # spent a discarded model call per turn.
            #
            # It also cost a measurement. Two reply handles per turn means two
            # `speech.start` messages, and the page's probe follows one utterance
            # at a time, so the discarded reply's ending switched the probe off
            # for the one actually speaking: twelve of fourteen utterances in
            # that run were not measured at all.
            "preemptive_generation": {"enabled": False},
        },
    )

    sender = bench_io.Sender(ctx.room)
    speech = speech_mod.Speech(session, tts, log, sender, dwell_s=s.status_dwell_s)
    app = Assistant(protocol=proto, speech=speech, log=log, sender=sender)
    # The scheduler calls back into the assistant, and the assistant needs the
    # scheduler, so one of the two has to be attached after construction.
    app.attach_scheduler(timers_mod.AsyncScheduler(app.on_timer))

    _watch_turns(session, log, sender)
    await session.start(WetlabAgent(app), room=ctx.room)
    dropped = bench_io.subscribe(ctx.room, lambda msg: _on_page(log, msg))

    async def shutdown() -> None:
        app._scheduler.cancel_all()
        log.append("shutdown", dropped_packets=dropped())
        log.close()

    ctx.add_shutdown_callback(shutdown)

    await sender.send(
        "hello",
        protocol=proto.id,
        title=proto.title,
        # Shown on the page. The product's claim is that it reads a published
        # protocol as published, so where the text came from belongs on screen
        # next to it rather than only in the run header.
        citation=proto.citation,
        steps=len(proto.steps),
        model="mistv3",
        speaker=s.rime_speaker or "(default)",
    )

    # It greets and waits. Reading step one at someone who has just joined and
    # is still putting their gloves on is the behaviour of something that
    # cannot be talked to.
    #
    # Through the model, like everything else. There is no utterance in this
    # product that anything other than the model produces, so there is nothing
    # that can arrive on top of an answer or say a thing twice.
    await speech.via_model(
        f"Greet them. Say this is {proto.title}, that it has {len(proto.steps)} "
        "steps, and that they should say start when they are ready. Two "
        "sentences at most. Do not read the first step.",
        kind="greeting",
    )

    script = os.environ.get("WETLAB_SCRIPT")
    if script:
        from . import scripted

        asyncio.create_task(scripted.play(session, Path(script), log))


def _watch_turns(session, log, sender) -> None:
    """Timestamp both ends of a turn, and tell the page about an interruption.

    Claim 3.4 is the interval between the scientist starting to speak and the
    audio stopping, and neither end of that is knowable from inside a tool.
    The near end is `user_state_changed`. The far end is measured on the page,
    from its own probe, rather than from the agent's idea of playback, so the
    page has to be told an interruption happened.
    """
    speaking = {"agent": False}

    @session.on("user_state_changed")
    def _user(ev) -> None:
        log.append("user.state", state=ev.new_state)
        if ev.new_state == "speaking" and speaking["agent"]:
            log.append("speech.interrupted")
            asyncio.create_task(sender.send("speech.interrupted", reason="user_speaking"))

    @session.on("user_input_transcribed")
    def _heard(ev) -> None:
        """What the recogniser made of what the scientist said.

        Interim results go to the page and are not logged. They exist so the
        screen can show a sentence forming while it is being said, and a log row
        per partial would bury the run in half-sentences. The final one is
        logged, because afterwards "it answered the wrong question" and "it
        heard the wrong question" look identical without it.
        """
        text = (getattr(ev, "transcript", "") or "").strip()
        if not text:
            return
        final = bool(getattr(ev, "is_final", False))
        if final:
            log.append("user.transcript", text=text)
        asyncio.create_task(
            sender.send("transcript", role="scientist", text=text, final=final)
        )

    @session.on("agent_state_changed")
    def _agent(ev) -> None:
        speaking["agent"] = ev.new_state == "speaking"
        log.append("agent.state", state=ev.new_state)

    @session.on("speech_created")
    def _speech(ev) -> None:
        """The model's own replies, which do not go through speech.py.

        A reply the model composes is spoken by the framework straight out of
        the LLM stream. It is the most common kind of turn there is, and until
        this was added it left no trace in the log at all: no speech.start, so
        the page never began a probe utterance for it, so there was no
        listener-side timing for the turns that matter most to claim 3.2.
        """
        if ev.source != "generate_reply":
            return  # a tool's own speech, already recorded by speech.py
        log.append("model.reply.created")
        asyncio.create_task(_follow_reply(ev.speech_handle, log, sender))

    @session.on("error")
    def _error(ev) -> None:
        # A provider failing is the one thing with no fallback path now that
        # there is no pedal, so it is recorded rather than only logged by the
        # framework.
        log.append("provider_failure", error=str(getattr(ev, "error", ev)))


def _reply_content(handle) -> tuple[str, list[dict]]:
    """What the model said, and what it called, from the handle's chat items.

    The items only exist once the generation is done, which is why this is read
    after playout rather than when the reply is created. `text_content` is the
    framework's own accessor and strips the expressive markup it adds, so what
    comes back is what was spoken.
    """
    text: list[str] = []
    calls: list[dict] = []
    for item in getattr(handle, "chat_items", ()) or ():
        kind = getattr(item, "type", None)
        if kind == "message" and getattr(item, "role", None) == "assistant":
            with contextlib.suppress(Exception):
                content = item.text_content
                if content and content.strip():
                    text.append(content.strip())
        elif kind == "function_call":
            calls.append(
                {
                    "name": getattr(item, "name", ""),
                    "arguments": str(getattr(item, "arguments", ""))[:500],
                }
            )
    return " ".join(text), calls


async def _follow_reply(handle, log, sender) -> None:
    """Record a model-composed reply the way speech.py records a step.

    The page is told, so its probe measures the reply from the listener's side
    like everything else. The context id is the handle's own, which is not one
    of ours: nothing in the fence keys on it, and it exists so the probe has
    something to bracket the utterance with.

    What the model actually said is read afterwards, from the handle. The
    utterance is streamed out of the LLM, so at `speech.start` there is no text
    to record yet, and the first version of this logged an empty string and
    left it there. That made every reply in every run a blank: nothing showed
    whether a question was answered from the protocol or invented, and nothing
    showed which tool a turn had called.
    """
    context_id = f"m{id(handle):x}"
    log.append("speech.start", context_id=context_id, kind="reply", step_id=None, text="")
    with contextlib.suppress(Exception):
        await sender.send("speech.start", context_id=context_id, kind="reply", step_id=None)
    with contextlib.suppress(Exception):
        await handle.wait_for_playout()

    text, calls = _reply_content(handle)
    for call in calls:
        log.append("tool.called", context_id=context_id, **call)
    log.append(
        "speech.finished",
        context_id=context_id,
        kind="reply",
        step_id=None,
        text=text,
        tools=[c["name"] for c in calls],
        interrupted=bool(getattr(handle, "interrupted", False)),
        silent=False,
        played=not getattr(handle, "interrupted", False),
    )
    with contextlib.suppress(Exception):
        await sender.send("step.state", context_id=context_id, played=True)

    # Sent after playout for the same reason it is logged after playout: the
    # reply is streamed out of the model, so there is no text to send when it
    # starts. The screen therefore shows what it said once it has said it,
    # which is the honest ordering and not a caption running ahead of the voice.
    #
    # Markup stripped on the way out. `use_tts_aligned_transcript` makes this
    # the text as it left `tts_node`, which is after `markup.apply` has put
    # spell(PCR) into it. That is right for the log, which is recording what
    # Rime was handed, and wrong for a screen, where it is not a word anybody
    # said: the greeting came up reading "This is spell(PCR) Using Q5".
    if text:
        with contextlib.suppress(Exception):
            await sender.send(
                "transcript",
                role="assistant",
                text=protocol_mod.without_markup(text),
                final=True,
            )

    # The reply first, then the calls it made. The page hangs a tool call on the
    # turn it is currently showing, so sending the call first put it under the
    # previous reply: start_protocol appeared beneath the greeting, which is the
    # one turn that cannot have called it.
    for call in calls:
        with contextlib.suppress(Exception):
            await sender.send("tool.called", context_id=context_id, **call)


def _on_page(log, msg: dict) -> None:
    """Every page message is logged under its own type, fields and all.

    The probe's names are therefore load-bearing for the evidence: renaming one
    in probe.js renames the event in the log.
    """
    log.append(msg["type"], **{k: v for k, v in msg.items() if k != "type"})


if __name__ == "__main__":
    cli.run_app(server)
