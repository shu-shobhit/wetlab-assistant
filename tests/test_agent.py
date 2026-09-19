"""The worker: reading back a reply, choosing a protocol, and what the page is told.

The first group builds the framework's own chat items rather than stand-ins, so
a change to the discriminator strings or to `text_content` fails here rather
than in a live run. A model-composed reply is streamed straight out of the LLM,
so at the moment the utterance starts there is no text to log. The first version
recorded an empty string and never came back for the real one, which made every
reply in every run a blank: nothing showed whether a question had been answered
from the protocol or invented, and nothing showed which tool a turn had called.

The rest is what the agent sends to the page, which has its own way of going
wrong quietly. A message the page never receives looks exactly like a page that
does not react to it.
"""

import asyncio
from pathlib import Path

import pytest
from livekit.agents import llm

from wetlab import agent, bench_io, protocol, timers
from wetlab.agent import _reply_content, protocol_for_room

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


class Handle:
    """Only the attribute `_reply_content` reads."""

    def __init__(self, *items):
        self.chat_items = list(items)


def message(text: str, role: str = "assistant") -> llm.ChatMessage:
    return llm.ChatMessage(role=role, content=[text])


def call(name: str, arguments: str = "{}") -> llm.FunctionCall:
    return llm.FunctionCall(call_id=f"c_{name}", name=name, arguments=arguments)


def test_the_reply_text_is_recovered():
    text, calls = _reply_content(Handle(message("The annealing step is thirty seconds.")))
    assert text == "The annealing step is thirty seconds."
    assert calls == []


def test_a_tool_call_is_recovered_with_its_arguments():
    """Which tool ran matters as much as the words. A model narrating a move it
    never made and a model calling go_to_step look identical in the text."""
    _, calls = _reply_content(Handle(call("go_to_step", '{"number": 2}')))
    assert calls == [{"name": "go_to_step", "arguments": '{"number": 2}'}]


def test_text_and_calls_come_back_together():
    text, calls = _reply_content(
        Handle(call("look_up", '{"query": "annealing"}'), message("Sixty five degrees."))
    )
    assert text == "Sixty five degrees."
    assert [c["name"] for c in calls] == ["look_up"]


def test_several_messages_are_joined():
    text, _ = _reply_content(Handle(message("Going back."), message("Step two.")))
    assert text == "Going back. Step two."


def test_the_users_own_words_are_not_recorded_as_the_reply():
    """The handle carries the turn, and the turn contains what was asked. Only
    the assistant's half is what the assistant said."""
    text, _ = _reply_content(Handle(message("what pH?", role="user"), message("Not in the protocol.")))
    assert text == "Not in the protocol."


def test_an_empty_message_does_not_become_a_blank_reply():
    assert _reply_content(Handle(message("   "))) == ("", [])


def test_a_handle_with_no_items_is_not_an_error():
    """A reply cut before it produced anything still has to be logged."""
    assert _reply_content(Handle()) == ("", [])


def test_a_handle_without_chat_items_at_all_is_not_an_error():
    """Interruption paths hand back objects that never gained the attribute."""
    assert _reply_content(object()) == ("", [])


def test_a_long_argument_string_is_truncated():
    """A log line is read by a person. An argument blob is not worth a screen."""
    _, calls = _reply_content(Handle(call("look_up", '{"query": "' + "x" * 900 + '"}')))
    assert len(calls[0]["arguments"]) == 500


# --- which protocol a room asks for -------------------------------------------


def test_the_room_name_chooses_the_protocol():
    chosen, why = protocol_for_room("bench.addgene_transformation.k3f9", CORPUS, "neb_q5_m0492")
    assert chosen == "addgene_transformation"
    assert why == "chosen on the page"


def test_an_id_that_is_not_in_the_corpus_falls_back_to_the_default():
    """The name comes from the browser. Honouring an unknown id would put the
    agent on a different protocol from the one named on screen, and that is the
    one failure here that would not be obvious while it was happening."""
    for asked in ("no_such_protocol", "/etc/passwd", "NEB_Q5_M0492"):
        chosen, why = protocol_for_room(f"bench.{asked}.k3f9", CORPUS, "neb_q5_m0492")
        assert chosen == "neb_q5_m0492", asked
        assert "not in the corpus" in why, asked


def test_a_room_that_names_no_protocol_falls_back():
    """Headless runs and the scripted driver never touch the page, so a room
    named the old way has to keep working."""
    for name in ("bench", "bench-k3f9-ab12", "bench..k3f9", "", None):
        chosen, why = protocol_for_room(name, CORPUS, "neb_q5_m0492")
        assert chosen == "neb_q5_m0492", name
        assert "default" in why


def test_the_random_part_may_hold_anything():
    chosen, _ = protocol_for_room("bench.handwritten.a.b.c", CORPUS, "neb_q5_m0492")
    assert chosen == "handwritten"


# --- what the page is told ----------------------------------------------------


class FakeSender:
    """Records what was sent, through the same whitelist the real one uses.

    Not a bare recorder. A name that is not in AGENT_MESSAGES raises, and the
    one time that happened it raised inside the entrypoint and crashed the job
    on its first utterance. A fake that accepts anything is exactly the fake
    that lets the next one through.
    """

    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    async def send(self, type, **payload):
        bench_io.decode_agent(bench_io.encode(type, **payload))
        self.sent.append((type, payload))

    def only(self, type):
        return [payload for kind, payload in self.sent if kind == type]


class FakeLog:
    def __init__(self):
        self.rows = []

    def append(self, type, **payload):
        self.rows.append((type, payload))


@pytest.fixture
def app():
    proto = protocol.load(CORPUS, "neb_q5_m0492")
    sender = FakeSender()
    a = agent.Assistant(protocol=proto, speech=None, log=FakeLog(), sender=sender)
    a.attach_scheduler(timers.Scheduler(a.on_timer))
    return a


async def test_the_step_reaches_the_page_as_written_and_as_spoken(app):
    """The page rendered a heading over an empty body for a whole day.

    It painted the step from the `speech.start` a speaking tool emitted, and
    that went when tools stopped speaking. Nothing else carried the text, and a
    panel that is empty because a message stopped being sent looks exactly like
    a panel that has nothing to show yet.
    """
    said = await app.present_step()
    begin = app.sender.only("step.begin")[0]
    assert begin["number"] == 1 and begin["total"] == len(app.protocol.steps)
    assert begin["text"] == said
    assert begin["source"] == app.protocol.steps[0].text
    assert begin["text"].strip() and begin["source"].strip()


async def test_the_page_is_never_shown_rime_markup(app):
    """`spell(Q)` is an instruction to the speech engine, not a word. Anything
    outside the speech path gets the readable form."""
    for _ in range(len(app.protocol.steps)):
        await app.present_step()
        app.position.next()
    for begin in app.sender.only("step.begin"):
        assert "spell(" not in begin["text"]
        assert "spell(" not in begin["source"]


async def test_a_timer_dropped_by_an_advance_is_taken_off_the_screen(app):
    """The page counts down off `timers.started` and has no other way to learn
    that one ended early. Cancelling was logged and not sent, so the row kept
    counting for the rest of the session and only the log knew it was wrong."""
    app.position.goto(12)
    app.advance()  # finishes s12, which declares a thirty second timer
    assert [t.id for t, _ in app.remaining_timers()] == ["s12.timer1"]

    app.advance()  # now on s13, so s12's timer is stale
    await asyncio.sleep(0)  # the send is scheduled, not awaited

    assert app.remaining_timers() == []
    assert app.sender.only("timer.cancelled") == [
        {"timer_id": "s12.timer1", "reason": "advanced"}
    ]


async def test_a_timer_cancelled_by_name_is_taken_off_the_screen(app):
    app.start_ad_hoc_timer(600, "wash")
    assert app.cancel_timer("wash") is True
    await asyncio.sleep(0)
    assert app.sender.only("timer.cancelled") == [{"timer_id": "ad_hoc.1", "reason": "asked"}]


# --- what an advance is allowed to cancel -------------------------------------


async def test_a_timer_the_scientist_asked_for_survives_an_advance(app):
    """It belongs to no step, so the rule that finishing a step ends what came
    before it does not reach it. "Set a timer for ten minutes" followed by "next
    step" used to lose it, and said nothing about having done so."""
    app.start_ad_hoc_timer(600, "wash")
    for _ in range(3):
        app.advance()
    assert [t.label for t, _ in app.remaining_timers()] == ["wash"]


async def test_advancing_still_drops_a_timer_the_step_before_declared(app):
    app.position.goto(12)
    app.advance()  # finishes s12, which declares a thirty second timer
    assert [t.id for t, _ in app.remaining_timers()] == ["s12.timer1"]
    app.advance()  # now out of s13, so s12's timer is stale
    assert app.remaining_timers() == []


async def test_a_step_keeps_its_own_timer_across_the_advance_that_started_it(app):
    """The timer starts as the step finishes, so the advance that starts it must
    not be the advance that ends it."""
    app.position.goto(12)
    app.advance()
    assert [t.id for t, _ in app.remaining_timers()] == ["s12.timer1"]


async def test_staleness_is_not_decided_by_a_prefix_of_the_timer_id(app):
    """Step ids run s1 to s17, so "s12.timer1" starts with "s1". Matching on the
    id let a timer from step twelve survive an advance out of step one, which is
    reachable by going back and then saying you are done."""
    app.position.goto(12)
    app.advance()  # s12.timer1 is now running, pointer on s13
    app.position.goto(1)
    app.advance()  # out of s1, which declared nothing
    assert app.remaining_timers() == []


async def test_cancelling_a_timer_that_is_not_running_tells_the_page_nothing(app):
    assert app.cancel_timer("no such timer") is False
    await asyncio.sleep(0)
    assert app.sender.only("timer.cancelled") == []


# --- ending the protocol ------------------------------------------------------


async def test_ending_the_protocol_stops_every_timer(app):
    """The reason this is a tool and not a sentence.

    Asked to end the protocol the model said "okay, ending the protocol here at
    step twelve", called nothing, and a twenty minute incubation carried on
    counting at a bench nobody was standing at.
    """
    app.position.goto(12)
    app.advance()  # s12 declares a thirty second timer
    app.start_ad_hoc_timer(600, "wash")
    assert len(app.remaining_timers()) == 2

    said = app.finish()
    await asyncio.sleep(0)

    assert app.remaining_timers() == []
    assert "stopped 2 running timers" in said
    assert {p["reason"] for p in app.sender.only("timer.cancelled")} == {"finished"}


async def test_ending_tells_the_page_where_it_ended(app):
    app.position.goto(4)
    app.finish()
    await asyncio.sleep(0)
    assert app.sender.only("protocol.finished") == [
        {"number": 4, "total": app.position.total, "timers_stopped": 0}
    ]


async def test_ending_with_nothing_running_says_so_rather_than_counting_zero(app):
    assert "no timers were running" in app.finish()


async def test_the_model_is_told_the_protocol_ended(app):
    app.position.started = True
    app.finish()
    assert "ended" in app.context_block()


async def test_reading_a_step_takes_it_out_of_the_ended_state(app):
    """Ending refuses nothing, like everything else here. Someone who says they
    are done and then asks for step one has changed their mind, and that is not
    a thing to argue with."""
    app.finish()
    assert app.position.ended is True
    await app.present_step()
    assert app.position.ended is False
    assert "ended" not in app.context_block()


# --- how the model is configured ---------------------------------------------


def _settings(**over):
    import dataclasses
    from wetlab import settings as settings_mod

    base = settings_mod.Settings(
        rime_api_key="r", openrouter_api_key="o", livekit_url="ws://x",
        livekit_api_key="k", livekit_api_secret="s", rime_speaker="luna",
        llm_model="z-ai/glm-5.3-flash", llm_reasoning=None, llm_provider="",
        stt_model="deepgram/nova-3",
        runs_dir=Path("runs"), corpus_dir=CORPUS, protocol_id="neb_q5_m0492",
        room_name="bench", probe_interval_ms=200, status_dwell_s=0.4,
    )
    return dataclasses.replace(base, **over)


def _body(**over):
    return agent.build_llm(_settings(**over))._opts.extra_body


def test_nothing_is_said_about_reasoning_unless_it_was_configured():
    """A model that does not reason must not be sent a field about reasoning,
    and one that does keeps its own default until somebody chooses otherwise."""
    assert "reasoning" not in _body(llm_reasoning=None)


def test_a_reasoning_level_is_sent_as_an_effort():
    """Not as a switch. `z-ai/glm-5.3-flash` answers 400 to `enabled: false`,
    "Reasoning is mandatory for this endpoint and cannot be disabled", and
    answers `effort: minimal` with zero reasoning tokens."""
    assert _body(llm_reasoning="minimal")["reasoning"] == {"effort": "minimal"}
    assert _body(llm_reasoning="high")["reasoning"] == {"effort": "high"}


def test_off_asks_for_no_reasoning_at_all():
    assert _body(llm_reasoning="off")["reasoning"] == {"enabled": False}


def test_the_model_is_reached_through_openrouter():
    llm = agent.build_llm(_settings())
    assert llm._opts.model == "z-ai/glm-5.3-flash"
    assert str(llm._client.base_url).rstrip("/") == agent.OPENROUTER_URL


def test_no_provider_is_named_unless_one_was_chosen():
    assert "provider" not in _body(llm_provider="")


def test_the_named_providers_are_tried_in_order_and_nothing_else_is():
    """Routing left alone balances price against speed, and the cheap hosts are
    the slow ones: the same model reached its first token in 0.84 s on one
    provider and 1.83 s routed, and the routed one dropped a tool call.

    Two names rather than one, because a single pin turns a provider's 429 into
    a failed turn. Fallbacks stay off so it can never leave the list."""
    body = _body(llm_provider="baidu/fp8, alibaba/fp8")
    assert body["provider"] == {"order": ["baidu/fp8", "alibaba/fp8"], "allow_fallbacks": False}


def test_a_provider_list_survives_spacing():
    assert _body(llm_provider=" wafer ,, makora ")["provider"]["order"] == ["wafer", "makora"]
