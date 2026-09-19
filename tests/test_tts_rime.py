import asyncio
from contextlib import asynccontextmanager

import aiohttp
import pytest
from livekit.agents import tts as tts_mod

from fakes.rime_ws import FakeRime, silence

from wetlab import tts_rime

SR = 22050


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


@pytest.fixture
def rime(session):
    """Start a fake /ws3 and point a FencedRimeTTS at it.

    Teardown order matters: the TTS pools its socket for 300 seconds, so the
    server cannot shut down until the client has let go of it.
    """

    @asynccontextmanager
    async def build(**fake_kwargs):
        fake = FakeRime(**fake_kwargs)
        await fake.start()
        tts = tts_rime.FencedRimeTTS(
            base_url=fake.base_url,
            model="mistv3",
            speaker="alexis",
            lang="eng",
            sample_rate=SR,
            api_key="not-a-real-key",
            http_session=session,
        )
        try:
            yield fake, tts
        finally:
            await tts.aclose()
            await fake.stop()

    return build


async def speak(stream, text: str) -> None:
    """Push the text the way the framework does, in pieces."""
    for part in text.split(" "):
        stream.push_text(part + " ")
    stream.end_input()


async def drain(stream) -> list:
    return [event async for event in stream]


def audio_seconds(pcm: bytes | bytearray) -> float:
    return len(pcm) / (SR * 2)


# --- the utterance goes out whole ---------------------------------------------


async def test_text_pushed_in_pieces_is_sent_as_one_message(rime):
    async with rime() as (fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "Add zero point five microlitres.")
        await drain(stream)
        # The plugin's own stream sends one message per sentence. This one holds
        # the utterance together, because a step is one utterance.
        assert fake.texts_for("u1") == ["Add zero point five microlitres."]
        assert ("flush", "u1") in fake.operations()


async def test_the_agent_chooses_the_context_id(rime):
    async with rime() as (fake, tts):
        tts.reserve_context("step-7")
        stream = tts.stream()
        await speak(stream, "hello")
        events = await drain(stream)
        assert fake.texts_for("step-7")
        assert {e.segment_id for e in events} == {"step-7"}


async def test_an_unreserved_stream_still_gets_an_id(rime):
    async with rime() as (fake, tts):
        stream = tts.stream()
        await speak(stream, "hello")
        await drain(stream)
        sent = [m for m in fake.received if "text" in m]
        assert sent and sent[0]["contextId"]


# --- the fence ----------------------------------------------------------------


async def test_another_utterances_audio_never_reaches_this_one(rime):
    # A preempted step's audio keeps arriving on the pooled socket. The plugin
    # does not filter it, so it plays inside whatever is speaking next.
    async with rime(bleed_context="ghost") as (fake, tts):
        tts.reserve_context("u2")
        stream = tts.stream()
        await speak(stream, "hello")
        events = await drain(stream)

        assert {e.segment_id for e in events} == {"u2"}
        assert tts.stale_dropped["ghost"] > 0
        # The ghost sends its own `done` first. If that ended the stream, u2
        # would stop before its own audio had all arrived.
        assert audio_seconds(tts.audio["u2"]) == pytest.approx(fake.chunks * fake.chunk_ms / 1000)
        assert "ghost" not in tts.audio


async def test_cut_tells_rime_to_stop(rime):
    async with rime(chunks=20, stale_after_clear_s=0.15) as (fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "hello")

        task = asyncio.create_task(drain(stream))
        await asyncio.sleep(0.1)
        tts.cut("u1")
        await asyncio.wait_for(task, timeout=5)

        assert ("clear", "u1") in fake.operations()


async def test_nothing_reaches_the_speaker_after_a_cut(rime):
    # Rime does not stop when it is told to. Whatever it sends next is dropped
    # here, or it plays over the announcement that replaced it.
    async with rime(chunks=20, stale_after_clear_s=0.2) as (_fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "hello")

        task = asyncio.create_task(drain(stream))
        await asyncio.sleep(0.1)
        tts.cut("u1")
        at_cut = len(tts.audio["u1"])
        await asyncio.wait_for(task, timeout=5)

        assert at_cut > 0, "the test needs the cut to land mid-utterance"
        assert len(tts.audio["u1"]) == at_cut, "audio was captured after the cut"
        assert tts.stale_dropped["u1"] > 0, "the fake kept producing; the fence should have counted it"


async def test_a_cut_socket_is_not_returned_to_the_pool(rime):
    # The pool would hand a socket that is still producing to the next
    # utterance. The fence would catch it, but there is no reason to keep it.
    async with rime(chunks=20, stale_after_clear_s=0.1) as (fake, tts):
        tts.reserve_context("u1")
        first = tts.stream()
        await speak(first, "hello")
        task = asyncio.create_task(drain(first))
        await asyncio.sleep(0.1)
        tts.cut("u1")
        await asyncio.wait_for(task, timeout=5)

        tts.reserve_context("u2")
        second = tts.stream()
        await speak(second, "again")
        await drain(second)

        assert fake.connections == 2, "the cut socket was reused"


async def test_a_context_cut_before_it_is_spoken_is_never_sent(rime):
    # The window between reserving a context and reaching it is small and real:
    # a timer can fire in it. Nothing has gone to Rime yet, so the cut costs
    # nothing and the step is simply not spoken.
    async with rime() as (fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        tts.cut("u1")
        await speak(stream, "hello")
        assert await drain(stream) == []
        assert fake.received == []
        assert tts.audio["u1"] == bytearray()


async def test_a_cut_utterance_is_not_retried_under_a_new_context(rime):
    # The framework retries a stream that pushed no audio, replaying the text
    # into a fresh _run that takes a context id the agent never reserved. That
    # id is not in _live, so cut() cannot reach it: a step cut by a timer would
    # be spoken again over the announcement that cut it.
    async with rime(chunks=20, gap_s=0.05, stale_after_clear_s=0.05) as (fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "hello")
        task = asyncio.create_task(drain(stream))
        await asyncio.sleep(0.02)  # before the first chunk arrives
        tts.cut("u1")
        await asyncio.wait_for(task, timeout=5)
        await asyncio.sleep(0.4)  # longer than the retry interval

        spoken = {m["contextId"] for m in fake.received if "text" in m}
        assert spoken == {"u1"}, f"the utterance was re-sent under {spoken - {'u1'}}"
        assert audio_seconds(tts.audio["u1"]) == 0.0


# --- evidence and failures ----------------------------------------------------


async def test_the_pcm_is_kept_per_context_for_the_evidence_clips(rime):
    async with rime(chunks=3, chunk_ms=100) as (_fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "hello")
        await drain(stream)
        assert audio_seconds(tts.audio["u1"]) == pytest.approx(0.3)
        assert bytes(tts.audio["u1"]) == silence(300)


async def test_a_sample_split_between_two_chunks_is_never_handed_over_in_halves(
    rime, monkeypatch
):
    """The bug that made an utterance come out as noise.

    Rime divides its stream wherever it likes, so a chunk can end in the middle
    of a sixteen-bit sample. The emitter buffers what it is given and force
    flushes when synthesis is slower than realtime, which it usually is; a flush
    landing on a partial sample discards the whole buffer, and the count it
    discards is odd, so every sample after it is built from the top byte of one
    and the bottom byte of the next.

    Asserted at the push rather than on the frames that come out, because the
    corruption needs a flush to land at the wrong moment and nothing here can
    make the emitter's own timer fire on cue. What can be checked is the thing
    that makes the moment harmless: it is never holding half a sample.
    """
    pushed: list[int] = []
    original = tts_mod.AudioEmitter.push

    def spy(self, data):
        pushed.append(len(data))
        return original(self, data)

    monkeypatch.setattr(tts_mod.AudioEmitter, "push", spy)

    async with rime(chunks=3, chunk_ms=100, split_samples=True) as (fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "hello")
        await drain(stream)

        assert any(len(p) % 2 for p in fake.chunk_payloads()), "the fake split nothing"
        assert pushed, "nothing reached the emitter"
        assert all(n % 2 == 0 for n in pushed), f"partial sample pushed: {pushed}"

        # The evidence clip is the stream as Rime sent it, which is a different
        # question from what is safe to play, and it stays whole.
        assert bytes(tts.audio["u1"]) == b"".join(fake.chunk_payloads())


async def test_a_provider_error_is_raised_not_swallowed(rime):
    async with rime(error_message="speaker not found") as (_fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        await speak(stream, "hello")
        with pytest.raises(Exception, match="speaker not found"):
            await drain(stream)


async def test_empty_text_does_not_open_a_stream(rime):
    async with rime() as (fake, tts):
        tts.reserve_context("u1")
        stream = tts.stream()
        stream.push_text("   ")
        stream.end_input()
        assert await drain(stream) == []
        assert fake.received == []


async def test_presynthesis_returns_audio_for_the_context_it_was_given(rime):
    async with rime() as (fake, tts):
        frame = await tts.presynthesize("hold this for later", "held1")
        assert frame.samples_per_channel > 0
        assert tts.audio["held1"], "audio should be teed under the bound context"
        sent = [m for m in fake.received if m.get("text")]
        assert sent[0]["contextId"] == "held1"


async def test_presynthesis_does_not_take_a_reserved_context(rime):
    """The reserve queue is first in, first out, and pre-synthesis runs during
    the pause while the scientist works, which is exactly when they may speak.

    If pre-synthesis reserved its id like a normal utterance, a say() starting
    at the same moment would pop the id meant for the next step and the
    pre-synthesis would pop the say's. Every later utterance would then key on
    the wrong context, breaking the fence, the log and the evidence clips
    together. So the held stream binds its id and never touches the queue.
    """
    async with rime() as (fake, tts):
        tts.reserve_context("u9")

        held, spoken = await asyncio.gather(
            tts.presynthesize("the next step", "held1"),
            _speak(tts, "the reply"),
        )

        assert held.samples_per_channel > 0
        assert spoken == "u9", "the reserved id must go to the ordinary utterance"
        assert not tts._reserved, "the queue should be empty, not shifted"

        by_text = {m["text"]: m["contextId"] for m in fake.received if m.get("text")}
        assert by_text["the next step"] == "held1"
        assert by_text["the reply"] == "u9"


async def _speak(tts, text):
    """Drive one ordinary utterance the way the framework does, and report the
    context id it was actually given."""
    stream = tts.stream()
    stream.push_text(text)
    stream.flush()
    stream.end_input()
    async for _ in stream:
        pass
    await stream.aclose()
    return stream._cid
