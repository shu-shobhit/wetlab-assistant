"""Scoring the round trip.

This decides the headline claim, so what it accepts and refuses matters more
than most things here. It has to be generous about form and strict about value:
twenty five and 25 and microliters and microlitres all mean the volume
survived; twenty does not, because that is a different volume.
"""

import pytest

from wetlab import scoring

SOURCE = "Add 25 µl of Q5 High-Fidelity 2X Master Mix and 0.5 µM primer at 98 °C."


def kinds(text):
    return {t.kind for t in scoring.tokens_in(text)}


def sources(text):
    return [t.source for t in scoring.tokens_in(text)]


def test_it_finds_the_numbers_units_and_acronyms():
    assert kinds(SOURCE) == {"number", "unit", "acronym"}
    assert "25" in sources(SOURCE)
    assert "µl" in sources(SOURCE)
    assert "Q5" in sources(SOURCE)


@pytest.mark.parametrize(
    "transcript",
    [
        # As words, which is what the prepared arm sends.
        "add twenty five microlitres of Q five high fidelity two X master mix "
        "and zero point five micromolar primer at ninety eight degrees celsius",
        # As digits with American spelling, which is what a recogniser often
        # writes. The value survived either way.
        "add 25 microliters of Q5 high fidelity 2X master mix and 0.5 "
        "micromolar primer at 98 degrees celsius",
    ],
)
def test_a_value_that_survived_scores_however_it_is_written(transcript):
    summary = scoring.summarise(scoring.score(SOURCE, transcript))
    assert scoring.percent(summary["all"]) == 100.0


def test_a_changed_value_is_not_recovered():
    """The failure the whole exercise is about. Twenty five microlitres became
    twenty, the sentence still reads perfectly, and it is the wrong volume."""
    transcript = (
        "add twenty microlitres of Q five master mix and zero point five "
        "micromolar primer at ninety eight degrees celsius"
    )
    scores = scoring.score(SOURCE, transcript)
    assert not next(s for s in scores if s.token.source == "25").recovered
    assert scoring.percent(scoring.summarise(scores)["numeric_only"]) < 100


def test_numbers_are_reported_separately_from_everything_else():
    """A rendering that gets the words right and the numbers wrong is worse
    than useless, so the claim is judged on numbers alone as well as overall."""
    transcript = "add twenty five you all of cue five two ex mix zero point five mew em at 98 see"
    summary = scoring.summarise(scoring.score(SOURCE, transcript))
    assert scoring.percent(summary["numeric_only"]) == 100.0
    assert scoring.percent(summary["all"]) < 100.0


def test_a_digit_running_into_a_letter_is_still_found():
    """A recogniser writes 2X and Q5 as single tokens. Without splitting them,
    the 2 in 2X could not be found and a recovered value scored as lost."""
    scores = scoring.score("Add 2X buffer.", "add 2X buffer")
    assert all(s.recovered for s in scores)


def test_a_single_letter_is_not_evidence():
    """Q5 scored on "q" alone matched almost any transcript, which made the
    acronym column meaningless."""
    token = next(t for t in scoring.tokens_in("Use Q5 mix.") if t.kind == "acronym")
    assert all(len(form.replace(" ", "")) > 1 for form in token.accepts)
    assert not scoring.score("Use Q5 mix.", "use the quick mix")[0].recovered


def test_an_acronym_spelled_letter_by_letter_counts():
    assert scoring.score("Add EDTA.", "add e d t a")[0].recovered
    assert scoring.score("Add EDTA.", "add EDTA")[0].recovered


def test_a_unit_that_was_not_heard_is_not_recovered():
    """Normalising drops non-ASCII, so µl collapses to l and °c to c.

    Left in as acceptable forms, those matched any transcript containing a
    stray letter. A raw reading came back as "zero point five l of ten mm",
    with the unit plainly lost, and scored as recovered.
    """
    scores = scoring.score("Add 0.5 µL.", "and zero point five l")
    unit = next(s for s in scores if s.token.kind == "unit")
    assert not unit.recovered
    assert next(s for s in scores if s.token.kind == "number").recovered


def test_the_symbol_still_counts_when_it_is_actually_written():
    assert scoring.score("Add 10 mM.", "add ten mm")[1].recovered


# --- ways this scored a correct reading as a failure -------------------------
#
# Every case below was a real miss in the round trip of 7 September, and in
# every one the value was spoken correctly and transcribed correctly. The tests
# that existed used only small numbers and plural units, so the scorer looked
# right while under-counting the arm it was measuring.


@pytest.mark.parametrize(
    "spoken",
    [
        "add two hundred and fifty microlitres",  # British, and how Rime says it
        "add two hundred fifty microlitres",  # American
    ],
)
def test_a_number_counts_with_or_without_the_and(spoken):
    """English inserts "and" after hundred and thousand, or does not, depending
    on where the speaker is from. Both are the same number, and only one of
    them was accepted."""
    scores = scoring.score("Add 250 µl.", spoken)
    assert next(s for s in scores if s.token.kind == "number").recovered


def test_the_and_is_offered_at_every_place_english_allows_one():
    forms = scoring._number_forms("1250")
    assert "one thousand two hundred fifty" in forms
    assert any("and" in f for f in forms)


def test_a_unit_counts_in_the_singular():
    """A unit is written singular whenever it qualifies a noun: "a ten
    centimetre plate". Only the plural was accepted, so the unit scored as
    lost in a reading where it was said and heard correctly."""
    scores = scoring.score("Plate onto a 10 cm plate.", "plate onto a ten centimetre plate")
    assert next(s for s in scores if s.token.kind == "unit").recovered


def test_a_singular_unit_counts_in_either_spelling():
    scores = scoring.score("Plate onto a 10 cm plate.", "plate onto a ten centimeter plate")
    assert next(s for s in scores if s.token.kind == "unit").recovered


def test_the_plural_is_not_matched_by_the_singular_inside_it():
    """Adding singulars must not make matching sloppy. Comparison is on whole
    words, so "microlitre" does not match inside "microlitres" by accident and
    a genuinely absent unit stays absent."""
    scores = scoring.score("Add 25 µl.", "add twenty five of the thing")
    assert not next(s for s in scores if s.token.kind == "unit").recovered


def test_an_empty_transcript_recovers_nothing():
    """Rime returning silence must not read as a perfect score."""
    summary = scoring.summarise(scoring.score(SOURCE, ""))
    assert summary["all"]["recovered"] == 0
    assert summary["all"]["total"] > 0


def test_a_step_with_no_notation_yields_no_tokens():
    assert scoring.tokens_in("Mix the reaction gently.") == ()
    assert scoring.percent(scoring.summarise([])["all"]) is None
