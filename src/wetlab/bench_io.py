"""The data channel between the agent and the bench page.

JSON objects with a `type`, published reliably on one topic. Architecture doc
section 5. Two things about it are worth stating, because both are easy to get
wrong and neither announces itself when it is wrong.

**Names are checked where they are written.** A mistyped message type is
otherwise invisible: the agent sends it, the page has no handler, nothing
happens and nothing complains. `encode` accepts only the names the agent sends
and `decode` only the names the page sends, so a typo raises at the call site.

**Nothing escapes the receive handler.** `data_received` fires inside the room's
event loop with a synchronous callback. An exception there is not confined to
one message, so a malformed packet and a raising handler are both caught and
counted rather than allowed out.

`speech.start` carries the text being spoken so the page can show it. It
carries no timings: the page's own probe is the only clock on the listener's
side, and it is what the interruption and first-audio figures are measured
from.

`transcript` is the exception to "the page renders what the agent sent about
itself": it also carries what the scientist said, as the recogniser heard it.
That is not for the assistant's benefit, it is for the person watching, and it
is the only way to see from outside whether an answer came from the protocol or
from a mishearing.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .events import _jsonable

TOPIC = "wetlab"

#: What the agent sends. Architecture doc section 5.
AGENT_MESSAGES = frozenset(
    {
        "hello",
        "step.begin",
        "speech.start",
        "speech.interrupted",
        "step.state",
        "timer.fired",
        "timers.started",
        "timer.cancelled",
        "tool.called",
        "transcript",
        "protocol.finished",
    }
)

#: What the page sends. There is no `input.pedal` any more. The product is
#: driven by talking, and a key that advances the protocol is a second way of
#: doing the one thing the whole design says is done by voice.
PAGE_MESSAGES = frozenset({"page.ready", "probe.onset", "probe.position", "probe.cut"})


def _encode(type: str, allowed: frozenset[str], payload: dict[str, Any]) -> bytes:
    if type not in allowed:
        raise ValueError(f"{type!r} is not a message this side sends; one of {sorted(allowed)}")
    record: dict[str, Any] = {"type": type}
    for key, value in payload.items():
        record[key] = _jsonable(value)
    return json.dumps(record).encode("utf-8")


def encode(type: str, **payload: Any) -> bytes:
    """An agent to page message."""
    return _encode(type, AGENT_MESSAGES, payload)


def encode_page(type: str, **payload: Any) -> bytes:
    """A page to agent message. Used by the tests and the harness; the real page
    builds these in JavaScript."""
    return _encode(type, PAGE_MESSAGES, payload)


def _decode(data: bytes, allowed: frozenset[str], sender: str) -> dict[str, Any]:
    try:
        msg = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise ValueError(f"not a JSON object: {type(msg).__name__}")
    kind = msg.get("type")
    if not kind:
        raise ValueError("message has no type")
    if kind not in allowed:
        raise ValueError(f"{kind!r} is not a message the {sender} sends")
    return msg


def decode(data: bytes) -> dict[str, Any]:
    """A message from the page. Raises ValueError on anything unusable."""
    return _decode(data, PAGE_MESSAGES, "page")


def decode_agent(data: bytes) -> dict[str, Any]:
    """A message from the agent. The real page decodes these in JavaScript; this
    is for the tests, the harness and anything replaying a run."""
    return _decode(data, AGENT_MESSAGES, "agent")


class Sender:
    """Agent to page. One instance per room."""

    def __init__(self, room) -> None:
        self._room = room

    async def send(self, type: str, **payload: Any) -> None:
        await self._room.local_participant.publish_data(
            encode(type, **payload), reliable=True, topic=TOPIC
        )


def subscribe(room, on_message: Callable[[dict[str, Any]], None]) -> Callable[[], int]:
    """Page to agent. `on_message` is called synchronously, so it must not await;
    schedule anything slow onto the loop yourself.

    Returns a callable giving the number of packets that could not be handled, so
    a run can report that it dropped some rather than looking clean.
    """
    dropped = [0]

    @room.on("data_received")
    def _handler(packet) -> None:  # rtc.DataPacket
        if getattr(packet, "topic", None) != TOPIC:
            return
        try:
            msg = decode(packet.data)
        except ValueError:
            dropped[0] += 1
            return
        try:
            on_message(msg)
        except Exception:
            # One bad message must not stop the channel for every later one.
            dropped[0] += 1

    return lambda: dropped[0]
