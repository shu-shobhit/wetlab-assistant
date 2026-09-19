"""Applying Rime's markup at the one place text becomes audio.

The markup used to live in the corpus text, so one string served both Rime,
which needs `spell(DNA)`, and the language model, which must never see it. The
model was shown it, repeated it, and Rime was handed the word "spell" to say.

Which terms are spelled is decided by the preparation prompt and read out of
the corpus it wrote. Nothing here decides it: this only applies a decision
already made, which is why there is no list of terms in this file that is not
derived from a protocol.
"""

from pathlib import Path

import pytest

from wetlab import markup, protocol

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"

TERMS = ("DNA", "dNTP", "LB", "Q")


async def as_stream(*chunks):
    for chunk in chunks:
        yield chunk


async def collect(stream):
    return "".join([piece async for piece in stream])


# --- which terms, and from where --------------------------------------------


def test_the_terms_come_from_what_the_preparation_pass_wrote():
    proto = protocol.load(CORPUS, "neb_q5_m0492")
    terms = markup.spelled_terms(s.speech for s in proto.spoken.values())
    assert "DNA" in terms
    assert all("spell(" not in t for t in terms)


def test_a_term_containing_another_is_offered_first():
    """Longest first, or the inner term is wrapped and the outer one never
    matches what is left."""
    terms = markup.spelled_terms(["spell(DNA) and spell(dNTP)"])
    assert list(terms) == sorted(terms, key=len, reverse=True)


def test_a_corpus_with_nothing_spelled_yields_nothing():
    assert markup.spelled_terms(["Mix the reaction gently."]) == ()


# --- applying it -------------------------------------------------------------


def test_a_known_term_is_wrapped():
    assert markup.apply("Add the DNA now.", TERMS) == "Add the spell(DNA) now."


def test_a_term_the_model_wrote_itself_is_wrapped():
    """The reason this moved out of the corpus. Under the old arrangement only
    text copied verbatim from a step carried the markup, so a reagent the model
    named in its own sentence was read as a word."""
    said = "You will need the LB plates from the fridge."
    assert markup.apply(said, TERMS) == "You will need the spell(LB) plates from the fridge."


def test_an_unknown_word_is_left_alone():
    assert markup.apply("Mix the reaction gently.", TERMS) == "Mix the reaction gently."


def test_a_term_inside_a_longer_word_is_not_wrapped():
    assert markup.apply("The DNAase enzyme.", TERMS) == "The DNAase enzyme."


def test_something_already_wrapped_is_not_wrapped_twice():
    """Twice is worse than not at all: Rime reads the outer one as the word."""
    assert markup.apply("Add spell(DNA) now.", TERMS) == "Add spell(DNA) now."


def test_the_corpus_capitalisation_is_what_gets_emitted():
    """A protocol writes dNTP with a lowercase d, and that is the term."""
    assert markup.apply("Add DNTP mix.", TERMS) == "Add spell(dNTP) mix."


def test_a_single_letter_term_is_wrapped_when_it_stands_alone():
    assert markup.apply("Add Q five master mix.", TERMS) == "Add spell(Q) five master mix."


def test_a_term_glued_to_a_number_is_wrapped():
    """The defect this boundary was widened for.

    "Q5" went to Rime whole, and Rime reads a Q in front of a number as the sign
    for the Guatemalan quetzal, so the protocol's own name was spoken as an
    amount of money. A digit is not a letter and must not fence a term off.
    """
    assert markup.apply("Add Q5 master mix.", TERMS) == "Add spell(Q) 5 master mix."


def test_a_number_glued_in_front_of_a_term_is_wrapped():
    assert markup.apply("A 2X master mix.", ("X",)) == "A 2 spell(X) master mix."


def test_a_letter_still_fences_a_term_off_on_either_side():
    """Widening for digits must not widen for letters. "DNAase" is one word."""
    assert markup.apply("The DNAase in aDNA samples.", TERMS) == "The DNAase in aDNA samples."


def test_the_greeting_that_reached_a_listener_is_wrapped():
    """Taken from `runs/evidence/response_time_deepseek/`, the first
    `speech.finished` of the run.

    Four models over four runs all copied `Q5` out of the raw title in the
    greeting instruction, so this is the text a listener was actually sent.
    """
    said = "This is the protocol for spell(PCR) Using Q5 High-Fidelity 2X Master Mix"
    out = markup.apply(said, ("PCR", "Q", "X"))
    assert "Q5" not in out
    assert out == (
        "This is the protocol for spell(PCR) Using spell(Q) 5 "
        "High-Fidelity 2 spell(X) Master Mix"
    )


def test_empty_text_is_not_an_error():
    assert markup.apply("", TERMS) == ""


# --- applying it across a stream ---------------------------------------------


async def test_a_term_split_across_two_chunks_is_still_wrapped():
    """The model's text arrives in pieces that do not respect word boundaries.
    Transforming each piece as it came would miss exactly the terms this
    exists to catch."""
    out = await collect(markup.stream(as_stream("Add the D", "NA now."), TERMS))
    assert out == "Add the spell(DNA) now."


async def test_a_term_split_across_three_chunks_is_still_wrapped():
    out = await collect(markup.stream(as_stream("Use ", "dN", "TP", " mix."), TERMS))
    assert out == "Use spell(dNTP) mix."


async def test_a_term_at_the_very_end_of_a_stream_is_wrapped():
    """The tail is held back until the stream ends, or the last word of every
    utterance would go through unmarked."""
    out = await collect(markup.stream(as_stream("Now add the ", "DNA"), TERMS))
    assert out == "Now add the spell(DNA)"


async def test_the_stream_preserves_everything_it_is_given():
    text = "Mix the reaction gently, then spin it down."
    out = await collect(markup.stream(as_stream(*text), TERMS))
    assert out == text


async def test_an_empty_stream_yields_nothing():
    assert await collect(markup.stream(as_stream(), TERMS)) == ""


@pytest.mark.parametrize("protocol_id", ["neb_q5_m0492", "addgene_transformation"])
async def test_every_prepared_step_survives_a_round_trip(protocol_id):
    """Strip the markup the preparation pass wrote, apply it again from the
    term list, and get back what was authored. If these disagree, the runtime
    is saying something different from what was reviewed and committed."""
    proto = protocol.load(CORPUS, protocol_id)
    terms = markup.spelled_terms(s.speech for s in proto.spoken.values())
    for step in proto.steps:
        authored = proto.speech_for(step)
        assert markup.apply(proto.readable_for(step), terms) == authored, step.id


# --- the seam where it is applied --------------------------------------------


async def test_tts_node_applies_the_markup_to_what_reaches_the_engine():
    """The whole point of the seam. What the model wrote is plain; what the
    speech engine is handed carries the markup, and nothing between the two
    can lose it."""
    from wetlab import agent as agent_mod

    seen: list[str] = []

    class FakeDefault:
        @staticmethod
        async def tts_node(self, text, model_settings):
            async for piece in text:
                seen.append(piece)
            return
            yield  # pragma: no cover - makes this an async generator

    proto = protocol.load(CORPUS, "neb_q5_m0492")
    app = type("App", (), {"protocol": proto})()
    node = agent_mod.WetlabAgent.tts_node

    class Stub(agent_mod.WetlabAgent):
        def __init__(self):
            self._spelled = markup.spelled_terms(s.speech for s in proto.spoken.values())

    stub = Stub()
    original = agent_mod.Agent.default
    agent_mod.Agent.default = FakeDefault
    try:
        result = node(stub, as_stream("Add the DNA to the ", "tube."), None)
        async for _ in result:  # pragma: no cover - the fake yields nothing
            pass
    except TypeError:
        # The fake returns a coroutine rather than an iterator; awaiting it is
        # what drives the stream.
        pass
    finally:
        agent_mod.Agent.default = original

    assert "".join(seen) == "Add the spell(DNA) to the tube."
