"""A stand-in for Rime's /ws3, so the fence can be tested without the network.

It speaks the same events as the real endpoint (`chunk`, `timestamps`, `done`,
`error`) and records every message it receives, so a test can assert what was
sent as well as what came back.

Two of its modes exist to reproduce behaviour the real endpoint has and a naive
fake would not:

    bleed     chunks tagged with somebody else's contextId arrive in the middle
              of the stream. This is what a preempted utterance looks like on a
              pooled socket, and the plugin does not filter it.
    stale     after a `clear`, it keeps producing for a while. Rime does not
              stop instantly, which is the whole reason the fence exists.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field

from aiohttp import WSMsgType, web

SAMPLE_RATE = 22050
BYTES_PER_SAMPLE = 2


def silence(ms: int) -> bytes:
    return b"\x00" * int(SAMPLE_RATE * BYTES_PER_SAMPLE * ms / 1000)


@dataclass
class FakeRime:
    #: How many chunks one flush produces.
    chunks: int = 4
    #: Milliseconds of audio per chunk, and the delay between them, so a test can
    #: cut mid-stream and know there is still more to come.
    chunk_ms: int = 60
    gap_s: float = 0.02
    #: Emit chunks tagged with this contextId alongside the real ones.
    bleed_context: str | None = None
    #: Keep producing for this long after a `clear`.
    stale_after_clear_s: float = 0.0
    #: Fail the stream instead of finishing it.
    error_message: str | None = None
    #: Slice the audio at odd byte offsets, so a sixteen-bit sample is split
    #: across two chunks. Rime divides its stream wherever it likes, and a
    #: partial sample reaching the emitter is what turned a live utterance into
    #: noise: a flush landing on one discards the buffer, and the bytes it
    #: discards are an odd count, so everything after it is a byte out of step.
    split_samples: bool = False

    received: list[dict] = field(default_factory=list)
    connections: int = 0
    _runner: web.AppRunner | None = None
    _port: int = 0
    _open: set = field(default_factory=set)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    async def start(self) -> str:
        app = web.Application()
        app.router.add_get("/ws3", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]
        return self.base_url

    async def stop(self) -> None:
        # Close the sockets first. The client pools them for 300 seconds, so
        # cleanup() would sit waiting for handlers that are still reading from
        # a connection nobody intends to send anything else on.
        for ws in list(self._open):
            await ws.close()
        self._open.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def texts_for(self, context_id: str) -> list[str]:
        return [m["text"] for m in self.received if m.get("contextId") == context_id and "text" in m]

    def operations(self) -> list[tuple[str, str | None]]:
        return [
            (m["operation"], m.get("contextId"))
            for m in self.received
            if "operation" in m
        ]

    async def _handler(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._open.add(ws)
        cleared: set[str] = set()
        tasks: list[asyncio.Task] = []

        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            self.received.append(event)
            operation = event.get("operation")

            if operation == "flush":
                tasks.append(asyncio.create_task(self._produce(ws, event["contextId"], cleared)))
            elif operation == "clear":
                cleared.add(event["contextId"])
                if self.stale_after_clear_s:
                    tasks.append(
                        asyncio.create_task(self._stale(ws, event["contextId"]))
                    )
            elif operation == "eos":
                break

        for task in tasks:
            task.cancel()
        self._open.discard(ws)
        await ws.close()
        return ws

    def chunk_payloads(self) -> list[bytes]:
        """The pieces one flush sends, in order.

        By default each is a whole number of samples, which is the easy case.
        `split_samples` slices the same amount of audio at odd offsets instead,
        so every boundary between chunks falls inside a sample. The payload is a
        ramp rather than silence there, because a stream reassembled one byte
        out of step is still exactly the right length and silence cannot show
        the difference.
        """
        if not self.split_samples:
            return [silence(self.chunk_ms)] * self.chunks
        ramp = bytes(i % 251 for i in range(len(silence(self.chunk_ms * self.chunks))))
        size = (len(ramp) // self.chunks) | 1
        return [ramp[i : i + size] for i in range(0, len(ramp), size)]

    async def _send(self, ws: web.WebSocketResponse, payload: dict) -> None:
        if not ws.closed:
            await ws.send_str(json.dumps(payload))

    async def _produce(self, ws: web.WebSocketResponse, cid: str, cleared: set[str]) -> None:
        if self.error_message is not None:
            await self._send(ws, {"type": "error", "contextId": cid, "message": self.error_message})
            return

        for payload in self.chunk_payloads():
            if cid in cleared:
                return
            await asyncio.sleep(self.gap_s)
            if self.bleed_context:
                await self._send(
                    ws,
                    {
                        "type": "chunk",
                        "contextId": self.bleed_context,
                        "data": base64.b64encode(silence(self.chunk_ms)).decode(),
                    },
                )
            await self._send(
                ws,
                {
                    "type": "chunk",
                    "contextId": cid,
                    "data": base64.b64encode(payload).decode(),
                },
            )

        if cid in cleared:
            return
        total_s = self.chunks * self.chunk_ms / 1000
        await self._send(
            ws,
            {
                "type": "timestamps",
                "contextId": cid,
                "word_timestamps": {
                    "words": ["one", "two"],
                    "start": [0.0, total_s / 2],
                    "end": [total_s / 2, total_s],
                },
            },
        )
        if self.bleed_context:
            # A `done` for somebody else must not end this stream.
            await self._send(ws, {"type": "done", "contextId": self.bleed_context})
        await self._send(ws, {"type": "done", "contextId": cid})

    async def _stale(self, ws: web.WebSocketResponse, cid: str) -> None:
        """What Rime does after a clear: keeps going for a bit."""
        deadline = asyncio.get_running_loop().time() + self.stale_after_clear_s
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(self.gap_s)
            await self._send(
                ws,
                {
                    "type": "chunk",
                    "contextId": cid,
                    "data": base64.b64encode(silence(self.chunk_ms)).decode(),
                },
            )
        await self._send(ws, {"type": "done", "contextId": cid})
