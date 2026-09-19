import json

import pytest

from wetlab import bench_io


class FakeLocalParticipant:
    def __init__(self):
        self.published = []

    async def publish_data(self, payload, *, reliable=True, destination_identities=(), topic=""):
        self.published.append((payload, reliable, topic))


class FakeRoom:
    """Enough of rtc.Room for the two functions here: `on` and a local participant."""

    def __init__(self):
        self.local_participant = FakeLocalParticipant()
        self._handlers = {}

    def on(self, event, callback=None):
        def register(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return register(callback) if callback else register

    def emit(self, event, *args):
        for fn in self._handlers.get(event, []):
            fn(*args)


class FakePacket:
    def __init__(self, data, topic):
        self.data = data
        self.topic = topic
        self.participant = None
        self.kind = None


# --- the wire format ----------------------------------------------------------


def test_encode_decode_round_trip():
    raw = bench_io.encode("step.begin", step_id="s1", index=0)
    assert json.loads(raw)["type"] == "step.begin"
    assert bench_io.decode_agent(raw) == {"type": "step.begin", "step_id": "s1", "index": 0}
    page = bench_io.encode_page("probe.onset", context_id="u1")
    assert bench_io.decode(page) == {"type": "probe.onset", "context_id": "u1"}


def test_decode_rejects_an_object_with_no_type():
    with pytest.raises(ValueError, match="type"):
        bench_io.decode(b"{}")


def test_decode_rejects_junk():
    with pytest.raises(ValueError):
        bench_io.decode(b"not json")
    with pytest.raises(ValueError, match="object"):
        bench_io.decode(b"[1, 2]")


def test_a_mistyped_message_name_is_caught_where_it_is_written():
    # A typo here is otherwise invisible: the page just never reacts.
    with pytest.raises(ValueError, match="step.beign"):
        bench_io.encode("step.beign", step_id="s1")
    with pytest.raises(ValueError, match="probe.onset"):
        bench_io.encode("probe.onset")  # a page message, the agent never sends it


def test_decode_only_accepts_messages_the_page_sends():
    raw = json.dumps({"type": "step.begin"}).encode()
    with pytest.raises(ValueError, match="step.begin"):
        bench_io.decode(raw)


def test_payloads_json_cannot_take_are_converted_not_raised_on():
    raw = bench_io.encode("timers.started", ids=frozenset({"s1.timer1"}))
    assert bench_io.decode_agent(raw)["ids"] == ["s1.timer1"]


# --- the room --------------------------------------------------------------


async def test_sender_publishes_reliably_on_the_wetlab_topic():
    room = FakeRoom()
    await bench_io.Sender(room).send("hello", model="mistv3")
    payload, reliable, topic = room.local_participant.published[0]
    assert reliable is True and topic == bench_io.TOPIC
    assert bench_io.decode_agent(payload)["model"] == "mistv3"


def test_subscribe_decodes_and_dispatches():
    room = FakeRoom()
    seen = []
    bench_io.subscribe(room, seen.append)
    room.emit(
        "data_received",
        FakePacket(bench_io.encode_page("probe.onset", context_id="u1"), bench_io.TOPIC),
    )
    assert seen == [{"type": "probe.onset", "context_id": "u1"}]


def test_subscribe_ignores_other_topics():
    room = FakeRoom()
    seen = []
    bench_io.subscribe(room, seen.append)
    room.emit("data_received", FakePacket(bench_io.encode_page("page.ready"), "other"))
    room.emit("data_received", FakePacket(bench_io.encode_page("page.ready"), None))
    assert seen == []


def test_a_bad_packet_does_not_escape_the_handler():
    # The handler runs inside the room's event loop. An exception there takes
    # down more than one message.
    room = FakeRoom()
    seen = []
    bench_io.subscribe(room, seen.append)
    room.emit("data_received", FakePacket(b"not json", bench_io.TOPIC))
    room.emit("data_received", FakePacket(bench_io.encode_page("page.ready"), bench_io.TOPIC))
    assert [m["type"] for m in seen] == ["page.ready"]


def test_a_raising_callback_does_not_escape_the_handler():
    room = FakeRoom()
    seen = []

    def handler(msg):
        seen.append(msg)
        raise RuntimeError("boom")

    bench_io.subscribe(room, handler)
    room.emit("data_received", FakePacket(bench_io.encode_page("page.ready"), bench_io.TOPIC))
    assert len(seen) == 1


def test_the_two_sides_share_no_message_name():
    assert "speech.start" in bench_io.AGENT_MESSAGES
    assert "probe.onset" in bench_io.PAGE_MESSAGES
    assert not bench_io.AGENT_MESSAGES & bench_io.PAGE_MESSAGES


def test_the_page_is_told_when_a_timer_stops_early():
    """It counts down off `timers.started` and has no other way to find out.
    Cancelling was logged and not sent, so a timer dropped by an advance kept
    counting on screen for the rest of the session."""
    assert "timer.cancelled" in bench_io.AGENT_MESSAGES
    assert bench_io.decode_agent(
        bench_io.encode("timer.cancelled", timer_id="s12.timer1", reason="advanced")
    )["reason"] == "advanced"


def test_a_transcript_carries_both_sides():
    """The exception to the page rendering only what the agent sent about
    itself. What the scientist said is on screen for the person watching, and it
    is the only way to see from outside whether an answer came from the protocol
    or from a mishearing."""
    for role in ("scientist", "assistant"):
        got = bench_io.decode_agent(
            bench_io.encode("transcript", role=role, text="thirty seconds", final=True)
        )
        assert got["role"] == role and got["final"] is True


def test_there_is_no_pedal_message_left():
    """The design opens by saying there is no pedal and no key. A page message
    that advances the protocol would be a second way of doing the one thing
    that is supposed to be done by talking."""
    assert "input.pedal" not in bench_io.PAGE_MESSAGES
    assert not any("pedal" in name for name in bench_io.PAGE_MESSAGES)
