"""Speaking, and the order it happens in.

The framework orders speech itself, so these do not re-test that. They test the
three things it has no opinion about: audio held from before it was needed, a
status that is dropped when the answer beats it, and the silent-utterance rule.

The fake session records the order things were handed to it and how long each
took, because order is the property that failed in version 1.
"""

import asyncio

import pytest

from wetlab import speech


class FakeFrame:
    def __init__(self, samples=2048):
        self.samples_per_channel = samples


class FakeHandle:
    def __init__(self, play_s=0.0, interrupted=False):
        self._play_s = play_s
        self.interrupted = interrupted

    async def wait_for_playout(self):
        if self._play_s:
            await asyncio.sleep(self._play_s)


class FakeSession:
    """Records what it was asked to say, in order."""

    def __init__(self, play_s=0.0):
        self.said: list[tuple[str, bool]] = []
        self.play_s = play_s
        self.interrupt_next = False

    def say(self, text, *, allow_interruptions=True, add_to_chat_ctx=True, audio=None):
        self.said.append((text, audio is not None))
        handle = FakeHandle(self.play_s, interrupted=self.interrupt_next)
        self.interrupt_next = False
        return handle

    @property
    def texts(self):
        return [t for t, _ in self.said]


class FakeTTS:
    def __init__(self, fail=False, silent=False):
        self.audio: dict[str, bytearray] = {}
        self.reserved: list[str] = []
        self.presynthesised: list[str] = []
        self.fail = fail
        self.silent = silent

    def reserve_context(self, context_id):
        self.reserved.append(context_id)
        if not self.silent:
            self.audio.setdefault(context_id, bytearray(b"\x00\x00"))

    async def presynthesize(self, text, context_id):
        if self.fail:
            raise RuntimeError("rime is down")
        self.presynthesised.append(text)
        self.audio.setdefault(context_id, bytearray(b"\x00\x00"))
        return FakeFrame()


class FakeLog:
    def __init__(self):
        self.rows: list[tuple[str, dict]] = []

    def append(self, type, **payload):
        self.rows.append((type, payload))

    def types(self):
        return [t for t, _ in self.rows]


@pytest.fixture
def parts():
    session, tts, log = FakeSession(), FakeTTS(), FakeLog()
    return session, tts, log, speech.Speech(session, tts, log, dwell_s=0.05)


async def test_each_utterance_gets_its_own_context_id(parts):
    session, tts, log, sp = parts
    await sp.say("one", kind="reply")
    await sp.say("two", kind="reply")
    assert tts.reserved == ["u1", "u2"]


async def test_a_status_then_an_answer_stay_in_order(parts):
    session, tts, log, sp = parts
    sp.start_status("checking the annealing step")
    await asyncio.sleep(0.12)  # longer than the dwell, so the status is spoken
    await sp.drain_status()
    await sp.say("sixty five degrees", kind="reply")
    assert session.texts == ["checking the annealing step", "sixty five degrees"]


async def test_a_fast_tool_drops_its_status(parts):
    """An instant lookup followed immediately by its answer is padding."""
    session, tts, log, sp = parts
    sp.start_status("checking the annealing step")
    await asyncio.sleep(0.01)  # the tool returned well inside the dwell
    sp.cancel_status()
    await sp.say("sixty five degrees", kind="reply")
    assert session.texts == ["sixty five degrees"]
    assert "status.dropped" in log.types()


async def test_a_slow_tool_speaks_its_status(parts):
    session, tts, log, sp = parts
    sp.start_status("looking that up")
    await asyncio.sleep(0.12)
    sp.cancel_status()
    await sp.drain_status()
    assert "looking that up" in session.texts
    assert "status.dropped" not in log.types()


async def test_speaking_drops_a_pending_status(parts):
    """A status covers the gap before the assistant makes any sound.

    Found in the first live run: a speaking tool queued its step and then its
    own status behind it, so "Starting the protocol" arrived after the step it
    was announcing.
    """
    session, tts, log, sp = parts
    sp.start_status("starting the protocol")
    await sp.say("Assemble all reaction components on ice.", kind="step", step_id="s1")
    await asyncio.sleep(0.12)  # well past the dwell
    assert session.texts == ["Assemble all reaction components on ice."]
    assert "status.dropped" in log.types()


async def test_a_status_can_still_be_spoken_when_nothing_else_is(parts):
    """The drop must not fire on the status's own utterance."""
    session, tts, log, sp = parts
    sp.start_status("checking the annealing step")
    await asyncio.sleep(0.12)
    await sp.drain_status()
    assert session.texts == ["checking the annealing step"]


async def test_an_empty_status_is_not_spoken(parts):
    session, tts, log, sp = parts
    sp.start_status("   ")
    await asyncio.sleep(0.12)
    assert session.texts == []


async def test_an_interrupted_utterance_is_not_played(parts):
    session, tts, log, sp = parts
    session.interrupt_next = True
    assert await sp.say("a long step", kind="step", step_id="s1") is False


async def test_an_utterance_that_made_no_audio_is_not_played():
    """A stream can end cleanly having spoken nothing: the fence refuses a
    context cut before it reached the socket, and the framework returns without
    raising on a 499. Both come back as finished and not interrupted, and
    taking that at face value marks a step heard that nobody heard."""
    session, tts, log = FakeSession(), FakeTTS(silent=True), FakeLog()
    sp = speech.Speech(session, tts, log)
    assert await sp.say("a step", kind="step", step_id="s1") is False
    finished = [payload for type, payload in log.rows if type == "speech.finished"]
    assert finished[0]["silent"] is True
    assert finished[0]["played"] is False


async def test_speaking_reports_start_and_finish(parts):
    session, tts, log, sp = parts
    await sp.say("hello", kind="reply")
    assert log.types() == ["speech.start", "speech.finished"]


# --- speaking through the model ---------------------------------------------


class _Session:
    """Only what via_model touches."""

    def __init__(self):
        self.replies: list[str] = []

    def generate_reply(self, *, instructions):
        self.replies.append(instructions)
        return object()  # never awaited, so it needs no interface


class _Sender:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, type, **payload):
        self.sent.append(type)


async def test_asking_the_model_to_speak_records_the_request():
    """A timer expiring is safety relevant and the model can decline or
    reword, so what was asked for is logged separately from what came of it."""
    log = FakeLog()
    sp = speech.Speech(_Session(), tts=None, log=log)
    await sp.via_model("The timer for the incubation has finished.", kind="announce")
    assert "speech.requested" in log.types()


async def test_asking_the_model_to_speak_reaches_the_session():
    session = _Session()
    sp = speech.Speech(session, tts=None, log=FakeLog())
    await sp.via_model("Greet them.", kind="greeting")
    assert session.replies == ["Greet them."]


async def test_the_request_is_not_sent_to_the_page():
    """bench_io whitelists what the agent may send. Sending an unregistered
    message raised inside the entrypoint and crashed the job before the
    greeting or the scripted driver had run."""
    sender = _Sender()
    sp = speech.Speech(_Session(), tts=None, log=FakeLog(), sender=sender)
    await sp.via_model("Greet them.", kind="greeting")
    assert sender.sent == []


async def test_asking_does_not_wait_for_the_model_to_finish_speaking():
    """The caller for the greeting is the entrypoint, and the scripted driver
    starts on the line after it. Blocking here would stop the run."""
    import asyncio

    sp = speech.Speech(_Session(), tts=None, log=FakeLog())
    await asyncio.wait_for(sp.via_model("Greet them.", kind="greeting"), timeout=1.0)
