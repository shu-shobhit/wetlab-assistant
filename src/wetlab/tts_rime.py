"""Rime with a fence around every utterance.

The plugin's own stream has three properties this product cannot accept, all of
them read from `livekit/plugins/rime/tts.py` rather than assumed.

**It never filters inbound events by `contextId`.** Rime tags everything it sends
with the context it belongs to, and the plugin ignores the tag and pushes the
audio at whatever stream happens to be reading the socket. Sockets are pooled for
300 seconds, so the audio of a step that was cut can arrive inside the timer
announcement that replaced it. In this product that is a wrong quantity spoken
over a warning, which is the failure the whole thing exists to prevent.

**It never sends `clear`.** Interrupting cancels the reading task; Rime is not
told, and keeps synthesising to the end.

**It chooses its own context id.** The agent needs an id it picked, because the
fence, the log and the evidence clips are all keyed by it.

So: one utterance per stream under an id the agent reserves, every inbound event
checked against that id, a `clear` on the way out, and the socket dropped from the
pool rather than handed to the next utterance.

There is no timestamp ledger. Played state is whole-utterance (architecture 7.4)
and nothing here needs Rime's word timings. They are still forwarded to the
emitter, because `use_tts_aligned_transcript` uses them to caption the page, and a
caption being slightly wrong is a cosmetic problem rather than a safety one.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections import deque

import aiohttp
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice.io import TimedString
from livekit.plugins import rime
from livekit.plugins.rime.tts import NUM_CHANNELS
from livekit.plugins.rime.tts import SynthesizeStream as _RimeStream

#: One sixteen-bit sample per channel, which is the smallest run of bytes the
#: emitter can be given without leaving it holding part of one.
BYTES_PER_SAMPLE = 2 * NUM_CHANNELS


class FencedRimeTTS(rime.TTS):
    """Rime, with one utterance per context and a fence on the way in.

    `use_websocket` and `segment` are fixed: the fence needs the websocket
    protocol's context ids, and server-side segmentation splits on the period
    inside a decimal, which every step here has.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.pop("use_websocket", None)
        kwargs.pop("segment", None)
        super().__init__(use_websocket=True, segment="never", **kwargs)
        self._reserved: deque[str] = deque()
        self._live: dict[str, _FencedStream] = {}

        #: Raw PCM per context at Rime's native rate, teed before any resampling.
        #: This is where the evidence clips come from, so they are the bytes Rime
        #: sent rather than a second synthesis of the same text.
        self.audio: dict[str, bytearray] = {}
        #: Contexts that have been cut. Nothing tagged with one may be emitted.
        self.dead: set[str] = set()
        #: Events dropped by the fence, by the context they were tagged with.
        #: A non-zero count here is the measurement behind "zero stale speech".
        self.stale_dropped: dict[str, int] = {}

    def reserve_context(self, context_id: str) -> None:
        """Claim the id the next stream will use. Called before each `say()`."""
        self._reserved.append(context_id)

    def _take_context(self) -> str:
        return self._reserved.popleft() if self._reserved else utils.shortuuid("ctx_")

    def cut(self, context_id: str) -> None:
        """Stop an utterance. Safe to call from a synchronous handler.

        The order matters. Marking the context dead is what stops audio reaching
        the speaker, and it takes effect immediately. Telling Rime is a courtesy
        that saves it work and saves us bandwidth; it is not the mechanism, and
        it cannot be, because the message takes as long to arrive as the audio it
        is trying to stop.
        """
        self.dead.add(context_id)
        stream = self._live.get(context_id)
        if stream is not None:
            stream.request_clear()

    def _count_dropped(self, context_id: str) -> None:
        self.stale_dropped[context_id] = self.stale_dropped.get(context_id, 0) + 1

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> _FencedStream:
        s = _FencedStream(tts=self, conn_options=conn_options)
        self._streams.add(s)
        return s

    async def presynthesize(
        self,
        text: str,
        context_id: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> rtc.AudioFrame:
        """Synthesise now, play later. Used to have the next step ready.

        The context id is bound to the stream here rather than reserved, and
        that is the whole reason this method exists instead of a
        `reserve_context` and a normal `stream()`.

        `_reserved` is a queue and `_take_context` pops from the front, so the
        id a stream gets depends only on the order things were reserved in.
        That is safe for `session.say()`, where the reserve and the synthesis
        are adjacent. It is not safe here: this runs during the pause while the
        scientist works, which is exactly when they may say something, so a
        concurrent `say()` would pop the id reserved for the next step and this
        would pop theirs. Every utterance after that keys on the wrong context,
        which breaks the fence, the log and the evidence clips at once.

        The plugin's own one-shot `synthesize()` is not an option: it raises
        when `use_websocket=True`, and the websocket is where the context ids
        live. Everything the scientist hears comes from `/ws3` either way, so
        the evidence describes one Rime path.
        """
        stream = _FencedStream(tts=self, conn_options=conn_options, context_id=context_id)
        self._streams.add(stream)
        frames: list[rtc.AudioFrame] = []
        try:
            stream.push_text(text)
            stream.flush()
            stream.end_input()
            async for event in stream:
                frames.append(event.frame)
        finally:
            await stream.aclose()
        if not frames:
            raise APIConnectionError(f"no audio for held context {context_id}")
        return rtc.combine_audio_frames(frames)


class _FencedStream(_RimeStream):
    def __init__(
        self,
        *,
        tts: FencedRimeTTS,
        conn_options: APIConnectOptions,
        context_id: str | None = None,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: FencedRimeTTS = tts
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._cid: str = ""
        self._clear_requested = False
        #: Bytes of a sample that arrived split across two chunks, held until
        #: the rest of it does. See the push in `_receive`.
        self._residue = bytearray()
        #: Set only by presynthesize(), which cannot use the reserve queue
        #: without racing a concurrent say() for its id.
        self._bound_context = context_id

    def request_clear(self) -> None:
        self._clear_requested = True
        ws = self._ws
        if ws is None or ws.closed:
            return
        task = asyncio.create_task(self._send_clear(ws))
        # Nothing awaits this, so an exception in it would surface much later as
        # an unretrieved-task warning with no context.
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _send_clear(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        with contextlib.suppress(Exception):
            await ws.send_str(json.dumps({"operation": "clear", "contextId": self._cid}))

    async def _collect_text(self) -> str:
        """The whole utterance, as one string.

        The plugin tokenises into sentences and sends one message per sentence.
        A step is one utterance and Rime is told not to segment, so there is
        nothing to gain from splitting it and one thing to lose: every extra
        message is another place for a partial send to leave half a step spoken.
        """
        parts: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                continue
            parts.append(data)
        return "".join(parts).strip()

    @staticmethod
    def _cut_error(cid: str) -> APIStatusError:
        """The framework's own signal for a stream its caller ended on purpose.

        `tts.SynthesizeStream` treats a stream that pushed no audio as a failed
        synthesis and retries it, replaying the buffered text into a fresh
        `_run` that takes a **new context id**. The agent never reserved that id,
        so it is not in `_live`, `cut()` cannot reach it, and played state cannot
        see it: a step cut by a timer would be spoken a second time, over the
        announcement that cut it. The window is small and it is exactly when a
        preemption lands, because a cut in the first moments is the case that
        pushes no audio.

        Status 499, Client Closed Request, is the one code `tts.py` returns on
        without raising and without retrying. That is what a cut is.
        """
        return APIStatusError("utterance cut", status_code=499, request_id=cid, body=None)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        cid = self._bound_context or self._tts._take_context()
        self._cid = cid
        self._tts._live[cid] = self
        self._tts.audio.setdefault(cid, bytearray())

        output_emitter.initialize(
            request_id=cid,
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=cid)

        try:
            text = await self._collect_text()
            if not text:
                output_emitter.end_input()
                return
            if cid in self._tts.dead:
                # Cut between reserving the context and reaching it. Nothing has
                # been sent, so there is nothing to clear and nothing to speak.
                raise self._cut_error(cid)
            await self._synthesize(text, cid, output_emitter)
            if cid in self._tts.dead:
                raise self._cut_error(cid)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=cid, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(f"Rime ws error: {e}") from e
        finally:
            self._tts._live.pop(cid, None)
            self._ws = None

    async def _synthesize(self, text: str, cid: str, output_emitter: tts.AudioEmitter) -> None:
        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._ws = ws
            try:
                self._mark_started()
                await ws.send_str(json.dumps({"text": text, "contextId": cid}))
                await ws.send_str(json.dumps({"operation": "flush", "contextId": cid}))
                if self._clear_requested:
                    # Cut between reserving the context and sending it. Rime has
                    # the text now, so it still has to be told to drop it.
                    await self._send_clear(ws)
                await self._receive(ws, cid, output_emitter)
            finally:
                if self._clear_requested:
                    # A cut socket is still producing. Handing it back to the
                    # pool means the next utterance reads somebody else's audio
                    # off it, and the fence should not be the only thing
                    # standing between that and the speaker.
                    self._tts._pool.remove(ws)

    async def _receive(
        self, ws: aiohttp.ClientWebSocketResponse, cid: str, output_emitter: tts.AudioEmitter
    ) -> None:
        while True:
            msg = await ws.receive(timeout=self._conn_options.timeout)
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIStatusError("Rime ws closed unexpectedly", request_id=cid)
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"Rime ws error: {ws.exception()}")
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            event = json.loads(msg.data)
            event_cid = event.get("contextId")
            kind = event.get("type")

            # The fence. Anything tagged with somebody else's context is dropped
            # before the emitter, including their `done`, which would otherwise
            # end this utterance early.
            if event_cid != cid:
                if event_cid is not None:
                    self._tts._count_dropped(event_cid)
                continue

            if kind == "done":
                # Ends the stream whether or not the context was cut, so a cut
                # utterance still terminates instead of waiting for a timeout.
                output_emitter.end_input()
                return
            if kind == "error":
                raise APIError(f"Rime ws error: {event.get('message', '(no message)')}")

            if cid in self._tts.dead:
                # Cut. Rime has not stopped yet and may not for a while.
                self._tts._count_dropped(cid)
                continue

            if kind == "chunk":
                pcm = base64.b64decode(event["data"])
                # Everything Rime sent, partial sample and all. This is what the
                # evidence clips are written from, so it records the stream
                # rather than the part of it that was convenient to play.
                self._tts.audio[cid] += pcm
                # Whole samples only, and this is not tidiness.
                #
                # Rime splits its stream on whatever boundary it likes, so a
                # chunk can end in the middle of a sixteen-bit sample. The
                # emitter buffers what it is given, and when synthesis is slower
                # than realtime it force-flushes that buffer to start playback.
                # A flush landing on a partial sample discards the whole buffer
                # (`AudioByteStream.flush` returns nothing, and the caller then
                # clears it), and the bytes it discards are an odd count, so
                # every sample after that is assembled from the top byte of one
                # and the bottom byte of the next. That is not distortion, it is
                # noise, and it is what a live run produced when asked to end the
                # protocol: two of ten flushes in that run hit it.
                #
                # Holding the split sample back until the rest of it arrives
                # means the buffer can never be flushed mid-sample. At most one
                # sample's worth is left over at the end of a stream, and it is
                # dropped, because part of a sample is not audio.
                self._residue += pcm
                whole = len(self._residue) - len(self._residue) % BYTES_PER_SAMPLE
                if whole:
                    output_emitter.push(bytes(self._residue[:whole]))
                    del self._residue[:whole]
            elif kind == "timestamps":
                timings = event.get("word_timestamps") or {}
                words = [
                    TimedString(text=word + " ", start_time=start, end_time=end)
                    for word, start, end in zip(
                        timings.get("words") or [],
                        timings.get("start") or [],
                        timings.get("end") or [],
                        strict=False,
                    )
                ]
                if words:
                    output_emitter.push_timed_transcript(words)
