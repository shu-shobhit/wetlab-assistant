"""The tools, driven against a real Assistant with the speaking faked out.

The version this replaces tested one property above all: that `advance` took no
arguments, so a model that was confused or jailbroken could not talk its way
past the gate. There is no gate, so what is worth testing is different. It is
that the pointer ends up where the scientist asked, that a tool speaks the
prepared wording rather than letting the model paraphrase a value, and that
advancing does the three things it has to do at once.
"""

from pathlib import Path

import pytest

from wetlab import agent, protocol, speech as speech_mod, timers, tools

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


class FakeSpeech:
    """Records what would have been said, and plays nothing."""

    def __init__(self):
        self.said: list[tuple[str, str]] = []
        self.statuses: list[str] = []
        self.prepared: list[str] = []
        self.played_held: list[str] = []
        self.discarded = 0
        self.next_is_interrupted = False

    async def say(self, text, *, kind, step_id=None, allow_interruptions=True):
        self.said.append((kind, text))
        played, self.next_is_interrupted = not self.next_is_interrupted, False
        return played

    async def prepare(self, text, *, step_id=None):
        self.prepared.append(text)
        return speech_mod.Held(context_id=f"h{len(self.prepared)}", text=text, step_id=step_id, frame=None)

    async def say_held(self, held, *, kind="step"):
        self.played_held.append(held.text)
        self.said.append((kind, held.text))
        return True

    def discard(self, held):
        if held is not None:
            self.discarded += 1

    async def cue(self):
        self.said.append(("cue", speech_mod.CUE_TEXT))

    def start_status(self, text):
        if text and text.strip():
            self.statuses.append(text)

    def cancel_status(self):
        pass

    async def drain_status(self):
        pass

    @property
    def steps_read(self):
        return [t for k, t in self.said if k == "step"]


@pytest.fixture
def app():
    proto = protocol.load(CORPUS, "neb_q5_m0492")
    a = agent.Assistant(protocol=proto, speech=FakeSpeech(), log=_Log())
    a.attach_scheduler(timers.Scheduler(a.on_timer, clock=_Clock()))
    return a


class _Log:
    def __init__(self):
        self.rows = []

    def append(self, type, **payload):
        self.rows.append((type, payload))

    def types(self):
        return [t for t, _ in self.rows]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def toolset(app):
    return {t.info.name: t for t in tools.build_tools(app)}


def call(tool, **kwargs):
    return tool(None, **kwargs)


async def test_next_step_moves_and_reads(app, toolset):
    await call(toolset["start_protocol"], status="")
    await call(toolset["next_step"], status="")
    assert app.position.number == 2


async def test_next_step_at_the_end_says_so_rather_than_failing(app, toolset):
    app.position.goto(app.position.total)
    out = await call(toolset["next_step"], status="")
    assert "last step" in out
    assert app.position.number == app.position.total


async def test_previous_step_goes_back(app, toolset):
    app.position.goto(5)
    await call(toolset["previous_step"], status="")
    assert app.position.number == 4


async def test_go_to_a_step_that_does_not_exist_says_which_ones_do(app, toolset):
    """Clamped, not refused. Which step it went to is more use than an
    apology, and the scientist cannot see the screen to count."""
    total = len(app.protocol.steps)
    out = await call(toolset["go_to_step"], number=99, status="")
    assert "no step 99" in out
    assert str(total) in out
    assert app.position.number == total


async def test_look_up_finds_by_a_spoken_question_without_speaking(app, toolset):
    out = await call(toolset["look_up"], query="what temperature", status="checking")
    assert "step" in out
    assert app.speech.steps_read == [], "look_up must not read anything aloud"


async def test_look_up_says_which_steps_were_already_read(app, toolset):
    await call(toolset["start_protocol"], status="")
    out = await call(toolset["look_up"], step_numbers=[1, 2], status="")
    assert "already read aloud" in out
    assert out.count("already read aloud") == 1


async def test_look_up_finding_nothing_says_so(app, toolset):
    out = await call(toolset["look_up"], query="unicorns", status="")
    assert "nothing" in out


async def test_every_tool_takes_a_status_and_offers_it(app, toolset):
    """The status is what keeps a tool turn from being silence. A tool without
    one is a tool the model cannot narrate, so this is checked for all ten
    rather than for the ones that happen to be slow."""
    import inspect

    for name, tool in toolset.items():
        params = inspect.signature(tool.__wrapped__).parameters
        assert "status" in params, name
    await call(toolset["where_are_we"], status="checking where we are")
    assert app.speech.statuses == ["checking where we are"]


async def test_timers_can_be_started_listed_and_cancelled(app, toolset):
    await call(toolset["start_timer"], seconds=30, label="the incubation", status="")
    assert "the incubation" in await call(toolset["list_timers"], status="")
    assert await call(toolset["cancel_timer"], label="incubation", status="") == "cancelled"
    assert await call(toolset["list_timers"], status="") == "no timers running"


async def test_cancelling_a_timer_that_is_not_running_says_so(app, toolset):
    out = await call(toolset["cancel_timer"], label="nothing", status="")
    assert "no timer matching" in out


async def test_a_timer_needs_a_positive_length(app, toolset):
    from livekit.agents import ToolError

    with pytest.raises(ToolError):
        await call(toolset["start_timer"], seconds=0, label="nothing", status="")


async def test_where_are_we_gives_the_number_and_the_total(app, toolset):
    app.position.goto(3)
    out = await call(toolset["where_are_we"], status="")
    assert f"step 3 of {len(app.protocol.steps)}" in out


async def test_no_tool_can_refuse(app, toolset):
    """The word that used to come back from advance."""
    for name in ("start_protocol", "read_current", "next_step", "previous_step"):
        out = await call(toolset[name], status="")
        assert "blocked" not in out.lower(), name


# --- what the model is allowed to see, and what it is told it has done -------


async def test_a_tool_that_reads_nothing_does_not_claim_to_have(app, toolset):
    """where_are_we answers a question. It must not tell the model it has
    spoken, or the model would stop answering."""
    out = await call(toolset["where_are_we"], status="")
    assert "just read this aloud" not in out


async def test_the_model_is_never_shown_the_speech_engines_markup(app, toolset):
    """spell(Q) is an instruction to Rime, not a word. It reached the model,
    the model repeated it, and Rime was handed "spell Q" to say."""
    spelled = [
        s for s in app.protocol.steps if "spell(" in app.protocol.speech_for(s)
    ]
    assert spelled, "the corpus should contain at least one spelled term"
    app.position.goto(spelled[0].number)
    assert "spell(" not in await call(toolset["where_are_we"], status="")
    assert "spell(" not in app.context_block()
    assert "spell(" not in app.spoken_text(spelled[0].id)


async def test_a_lookup_hands_the_model_readable_text(app, toolset):
    out = await call(toolset["look_up"], query="master mix", status="")
    assert "spell(" not in out


# --- tools change state and return words; they never speak -------------------


async def test_a_moving_tool_returns_the_step_and_speaks_nothing(app, toolset):
    """The refactor of 7 September. A tool used to read the step aloud and the
    framework then spoke the model's reply about the same turn, so the
    scientist heard every step twice. Output as a side effect of moving a
    pointer was the cause, so the tool returns words and says nothing."""
    out = await call(toolset["start_protocol"], status="")
    assert out == app.protocol.readable_for(app.protocol.steps[0])
    assert app.speech.said == [], "a tool must not speak"


async def test_every_moving_tool_returns_the_current_step(app, toolset):
    for name in ("start_protocol", "read_current", "next_step", "previous_step"):
        out = await call(toolset[name], status="")
        assert out == app.protocol.readable_for(app.position.current), name
        assert app.speech.said == [], name


async def test_the_returned_wording_is_the_prepared_one(app, toolset):
    """Not the source text. The whole point of the offline pass is that a
    number reaches the listener as words, and it does that by the model being
    handed the prepared wording rather than the notation."""
    numbered = next(s for s in app.protocol.steps if any(c.isdigit() for c in s.text))
    out = await call(toolset["go_to_step"], number=numbered.number, status="")
    assert not any(c.isdigit() for c in out), out


async def test_what_a_tool_returns_carries_no_markup(app, toolset):
    """Markup is applied in tts_node, after the model. Anything the model is
    given must be plain, because whatever it is shown it may say."""
    spelled = next(s for s in app.protocol.steps if "spell(" in app.protocol.speech_for(s))
    out = await call(toolset["go_to_step"], number=spelled.number, status="")
    assert "spell(" not in out


async def test_presenting_a_step_records_that_it_came_up(app, toolset):
    """look_up tells the model which steps have already been reached. It is
    the one thing left that reads this ledger."""
    await call(toolset["start_protocol"], status="")
    assert app.protocol.steps[0].id in app.position.read_aloud
    assert "step.presented" in app.log.types()
