"""Retrieval, scored against questions in the form they are actually spoken.

The questions here are the test. A bench scientist does not ask "what is the
duration of the annealing step", they ask "how long do I wait", and none of
"how", "long" or "wait" appears anywhere in a step that says 30 seconds. If
retrieval only works for questions phrased in the document's own words it does
not work, because the document is the thing the person cannot see.
"""

import pytest

from wetlab import render, retrieval

# The cited protocol, as text. Kept here rather than loaded so this test states
# what it is scoring against and does not change when the corpus does.
STEPS = [
    "Assemble all reaction components on ice.",
    "Add 25 µl of Q5 High-Fidelity 2X Master Mix to the reaction tube.",
    "Add 2.5 µl of 10 µM forward primer, for a final concentration of 0.5 µM.",
    "Add 2.5 µl of 10 µM reverse primer, for a final concentration of 0.5 µM.",
    "Add 1000 ng of template DNA to the reaction.",
    "Add nuclease-free water to a final volume of 50 µl.",
    "Mix the reaction gently, then collect the liquid at the bottom of the tube.",
    "Transfer the reaction to a thermocycler preheated to 98 °C.",
    "Run the initial denaturation at 98 °C for 30 seconds.",
    "Run 30 cycles. Denature at 98 °C for 10 seconds.",
    "Anneal at 65 °C for 30 seconds.",
    "Extend at 72 °C for 30 seconds per kilobase.",
    "Run a final extension at 72 °C for 2 minutes.",
    "Hold the reaction at 4 °C.",
]


@pytest.fixture
def index():
    return retrieval.Index(
        [
            retrieval.Entry(i + 1, f"s{i + 1}", text, render.terms(text))
            for i, text in enumerate(STEPS)
        ]
    )


# (question, the step numbers any of which is a right first answer)
SPOKEN_QUESTIONS = [
    # The load-bearing case. "temperature" appears in none of these fourteen
    # steps; they say 98 °C. Searching raw text for it returns nothing at all.
    ("what temperature did you mention", {8, 9, 10, 11, 12, 13, 14}),
    ("what was the temperature again", {8, 9, 10, 11, 12, 13, 14}),
    ("how hot does it get", {8, 9, 10, 11, 12, 13, 14}),
    # Likewise "how long": the steps say 30 seconds.
    ("how long do I wait", {9, 10, 11, 12, 13}),
    ("what was the duration", {9, 10, 11, 12, 13}),
    # And "how much".
    ("how much volume", {2, 3, 4, 6}),
    ("what amount of water", {6}),
    ("how many cycles", {10}),
    # Questions in the document's own words still have to work.
    ("what concentration of primer", {3, 4}),
    ("which step used the master mix", {2}),
    ("what did you say about ice", {1}),
    ("when do I add the template", {5}),
    ("what about the thermocycler", {8}),
]


@pytest.mark.parametrize("question,acceptable", SPOKEN_QUESTIONS)
def test_a_spoken_question_finds_the_right_step(index, question, acceptable):
    hits = index.search(query=question)
    assert hits, f"{question!r} found nothing"
    assert hits[0].number in acceptable, [h.number for h in hits]


OUT_OF_DOMAIN = [
    "what colour is the bench",
    "tell me about unicorns",
    "who won the cricket",
]


@pytest.mark.parametrize("question", OUT_OF_DOMAIN)
def test_a_question_about_nothing_in_the_protocol_returns_nothing(index, question):
    """Better to say it was not found than to hand the model an arbitrary step.

    Every step shares a word like "the" with every question, so candidacy that
    counted those would return the whole corpus for anything at all.
    """
    assert index.search(query=question) == ()


def test_terms_are_what_make_the_category_questions_work(index):
    """The same question against text alone, to show the terms are load-bearing."""
    bare = retrieval.Index(
        [retrieval.Entry(i + 1, f"s{i + 1}", t) for i, t in enumerate(STEPS)]
    )
    assert bare.search(query="what temperature did you mention") == ()
    assert index.search(query="what temperature did you mention") != ()


def test_step_numbers_come_back_in_the_order_asked(index):
    hits = index.search(steps=[9, 2])
    assert [h.number for h in hits] == [9, 2]
    assert hits[0].text == STEPS[8]


def test_a_step_number_out_of_range_is_skipped_not_an_error(index):
    assert [h.number for h in index.search(steps=[2, 99])] == [2]


def test_numbers_and_a_query_can_be_asked_for_together(index):
    hits = index.search(steps=[1], query="primer")
    assert hits[0].number == 1
    assert {h.number for h in hits} >= {1, 3}


def test_read_aloud_separates_what_was_heard_from_what_is_written(index):
    """The difference between "what temperature did you mention" and "what
    temperatures are in this protocol"."""
    hits = index.search(query="temperature", read_aloud=["s9"])
    heard = {h.number: h.read_aloud for h in hits}
    assert heard[9] is True
    assert all(v is False for k, v in heard.items() if k != 9)


def test_at_most_five_come_back(index):
    # Every step matches, so this is the cap and not the corpus running out.
    assert len(index.search(query="reaction °C add the", limit=5)) <= 5


def test_a_decimal_stays_one_token():
    """0.5 split on the point matches every step with a zero or a five in it."""
    assert retrieval.tokenise("0.5 µM") == ["0.5", "m"]


def test_a_plural_finds_the_singular(index):
    """The step says primer and the scientist says primers.

    Without folding, a question about the primers matched nothing at all in a
    protocol with two primer steps in it.
    """
    assert retrieval.tokenise("primers") == retrieval.tokenise("primer")
    hits = index.search(query="what did you say about the primers")
    assert {h.number for h in hits} >= {3, 4}


def test_a_phrase_term_is_searchable_word_by_word():
    """The offline pass writes some terms as phrases, and left whole those are
    single tokens no query word can ever equal."""
    idx = retrieval.Index(
        [retrieval.Entry(1, "s1", "Add the reagent.", ("nuclease free water", "master mix"))]
    )
    assert idx.search(query="water")
    assert idx.search(query="master")


def test_a_full_stop_is_not_part_of_the_word():
    assert "ice" in retrieval.tokenise("Assemble components on ice.")


def test_an_empty_index_does_not_explode():
    """BM25 divides by the average document length."""
    assert retrieval.Index([]).search(query="anything") == ()
