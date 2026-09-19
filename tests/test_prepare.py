"""Accepting or rejecting what the preparation pass wrote.

Two things do the accepting. Local rules that need no table cover the things a
table is actually good at: a digit is a digit, and a term is or is not a plain
word. A second model, from a different family than the one that wrote the text,
covers whether the values survived.

That split is the result of measurement rather than taste. The rule table that
used to do the value checking produced five false rejections and zero catches
over thirty prepared steps: every time it fired, the model was right and the
table was missing an entry for "mins", or for a fraction, or for picograms.

The verifier's own behaviour is checked against corrupted output in
tests/test_verifier_live.py, which needs a key. What is here is everything that
can be decided offline.
"""

import pytest

from wetlab import prepare

SOURCE = "Add 25 µl of Q5 High-Fidelity 2X Master Mix to the reaction tube."
GOOD = "Add twenty five microlitres of Q five High-Fidelity two X Master Mix to the reaction tube."
TERMS = ("volume", "microlitres", "master", "mix", "reagent")


# --- the local rules --------------------------------------------------------


def test_good_output_passes():
    assert prepare.check("s2", SOURCE, GOOD, TERMS).ok


def test_a_surviving_digit_is_caught():
    """A digit reaching Rime is a digit its own normaliser expands however it
    likes, which is the guessing this pass exists to remove. No table needed
    to know that, and this rule has never misfired."""
    result = prepare.check("s2", SOURCE, GOOD.replace("twenty five", "25"), TERMS)
    assert not result.ok
    assert any("digits left" in p for p in result.problems)


def test_an_empty_spoken_form_is_caught():
    """Silence read aloud is indistinguishable from a dead worker."""
    result = prepare.check("s2", SOURCE, "   ", TERMS)
    assert not result.ok
    assert any("empty" in p for p in result.problems)


@pytest.mark.parametrize(
    "terms,why",
    [
        (("volume",), "too few"),
        (tuple(f"word{i}" for i in range(20)), "too many"),
    ],
)
def test_the_term_count_is_bounded(terms, why):
    """Too few and a step cannot be found. Too many and BM25's scoring is
    diluted by words that describe every step equally."""
    assert not prepare.check("s2", SOURCE, GOOD, terms).ok, why


def test_markup_in_a_term_is_caught():
    """A term list feeds a keyword index; markup in one is useless. A live run
    produced an output listing spell(pH) as a term."""
    result = prepare.check("s2", SOURCE, GOOD, ("volume", "spell(pH)", "master", "mix"))
    assert not result.ok
    assert any("spell(pH)" in p for p in result.problems)


def test_an_uppercase_term_is_caught():
    assert not prepare.check("s2", SOURCE, GOOD, ("Volume", "master", "mix", "reagent")).ok


def test_the_local_rules_do_not_judge_values():
    """A changed value passes here. That is the verifier's job now, and saying
    so is the point of this test: nothing local pretends to have caught it."""
    corrupted = GOOD.replace("twenty five", "twenty")
    assert prepare.check("s2", SOURCE, corrupted, TERMS).ok


# --- reading the verifier's answer ------------------------------------------


def test_an_approval_is_no_problems():
    assert prepare.read_verdict('{"ok": true}') == []


def test_a_rejection_carries_its_reasons():
    problems = prepare.read_verdict('{"ok": false, "problems": ["volume changed", "unit dropped"]}')
    assert len(problems) == 2
    assert "volume changed" in problems[0]


def test_a_rejection_with_one_reason_as_a_string_still_reads():
    assert len(prepare.read_verdict('{"ok": false, "problems": "volume changed"}')) == 1


def test_a_rejection_with_no_reason_is_still_a_rejection():
    assert prepare.read_verdict('{"ok": false}') != []


@pytest.mark.parametrize("answer", ["not json at all", "[1, 2]", "", None])
def test_an_unreadable_verdict_is_a_problem_not_a_pass(answer):
    """Silence is not approval when the thing being approved gets read aloud
    to somebody who cannot check it."""
    assert prepare.read_verdict(answer) != []


def test_the_verifier_is_a_different_family_from_the_writer():
    """Two models sharing training and architecture share failure modes, and a
    checker that fails where the thing it checks fails is not a checker."""
    assert not prepare.VERIFIER_MODEL.startswith("z-ai/")
