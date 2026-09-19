"""The detector, over a table much larger than the corpus.

The corpus holds twenty distinct entity tokens. The detector's job is the
protocol nobody has uploaded yet, so the cases here are mostly notation the
corpus does not contain: ranges, ratios, scientific notation, negative
temperatures, compound units, chemical formulae, catalogue numbers.

Two invariants matter more than any single case and are asserted separately at
the bottom: no digit survives a render, and nothing is silently dropped.
"""

import re

import pytest

from wetlab import render


# (source, expected spoken form)
SPOKEN = [
    # The common case: a number and a unit.
    ("25 µl", "twenty five microlitres"),
    ("25 µL", "twenty five microlitres"),
    ("0.5 µL", "zero point five microlitres"),
    ("10 mM", "ten millimolar"),
    ("0.5 µM", "zero point five micromolar"),
    ("98 °C", "ninety eight degrees celsius"),
    ("30 s", "thirty seconds"),
    ("2 min", "two minutes"),
    ("1000 ng", "one thousand nanograms"),
    ("1 M", "one molar"),
    ("70%", "seventy percent"),
    # Units outside the table the span path ever needed.
    ("2 U/µl", "two units per microlitre"),
    ("100 V", "one hundred volts"),
    ("50 ng/µL", "fifty nanograms per microlitre"),
    ("37 °F", "thirty seven degrees fahrenheit"),
    ("500 bp", "five hundred base pairs"),
    # Negative temperatures, in all three dashes a protocol might use.
    ("-20 °C", "minus twenty degrees celsius"),
    ("−80 °C", "minus eighty degrees celsius"),
    # Ranges.
    ("55-65 °C", "fifty five to sixty five degrees celsius"),
    ("55 to 65 °C", "fifty five to sixty five degrees celsius"),
    ("10–15 min", "ten to fifteen minutes"),
    # Plural abbreviations, which the second protocol writes throughout.
    # Without "mins" in the table, "20-30 mins" did not match the range
    # pattern, fell through to two bare numbers, and the second of them was
    # read as *negative thirty*.
    ("20-30 mins", "twenty to thirty minutes"),
    ("30-60 secs", "thirty to sixty seconds"),
    ("250-1,000 µl", "two hundred fifty to one thousand microlitres"),
    # A range with nothing after it.
    ("1 - 5", "one to five"),
    ("10 pg", "ten picograms"),
    # Fractions. A protocol says to submerge the bottom half of a tube, not
    # the bottom one-slash-two of it, and Rime reads a slash as "slash".
    ("1/2", "one half"),
    ("2/3", "two thirds"),
    ("3/4", "three quarters"),
    ("5/8", "five over eight"),
    # Ratios and multipliers.
    ("1:10", "one to ten"),
    ("1:1", "one to one"),
    ("2X", "two X"),
    # Relative centrifugal force, which is not a mass in grams.
    ("10,000 × g", "ten thousand times g"),
    ("300 × g", "three hundred times g"),
    # Scientific notation.
    ("1.5 × 10⁶", "one point five times ten to the power six"),
    ("2 × 10^3", "two times ten to the power three"),
    # Acidity.
    ("pH 8.0", "p H eight point zero"),
    ("pH 7", "p H seven"),
    # Oligonucleotide ends.
    ("5'", "five prime"),
    # Acronyms are spelled letter by letter.
    ("DNA", "spell(DNA)"),
    ("EDTA", "spell(EDTA)"),
    ("PCR", "spell(PCR)"),
    # A leading lowercase letter is a chemistry prefix, not a word. This is the
    # one acronym the corpus actually contains and the obvious isupper() test
    # misses it.
    ("dNTP", "spell(dNTP)"),
    # A letter and a digit is a product name, said as its parts.
    ("Q5", "Q five"),
    # A catalogue number is a label, so its digits are read one at a time. Read
    # as a cardinal it loses the leading zero and a listener writes down a
    # different part number.
    ("M0492", "M zero four nine two"),
]


@pytest.mark.parametrize("source,expected", SPOKEN)
def test_detector_speaks_notation_correctly(source, expected):
    assert render.render(source) == expected


# (source, terms that must be present)
TERMS = [
    # The point of the whole mechanism: "temperature" appears nowhere in the
    # text, so a keyword search for it over raw text returns nothing. Whatever
    # recognises the degree sign in order to speak it supplies the term.
    ("98 °C", {"temperature", "98"}),
    ("25 µl", {"volume", "microlitres", "25"}),
    ("0.5 µM", {"concentration", "micromolar"}),
    ("30 s", {"time", "seconds", "30"}),
    ("1000 ng", {"mass", "nanograms"}),
    ("10,000 × g", {"force", "speed", "centrifuge"}),
    ("100 V", {"voltage", "volts"}),
    ("2 U/µl", {"activity"}),
    ("pH 8.0", {"ph", "acidity"}),
    ("1:10", {"ratio", "dilution"}),
    ("55-65 °C", {"range", "temperature"}),
    ("dNTP", {"acronym", "reagent", "dntp"}),
    ("M0492", {"catalogue", "part", "m0492"}),
    ("MgCl2", {"formula", "reagent", "mgcl2"}),
]


@pytest.mark.parametrize("source,expected", TERMS)
def test_detector_supplies_index_terms(source, expected):
    assert expected <= set(render.terms(source))


def test_terms_are_plain_lowercase_words():
    """A term list feeds a keyword index. Markup in it is useless.

    The live test of the offline pass found one output listing `spell(pH)` as
    a term, which is why this is asserted rather than assumed.
    """
    text = "Add 25 µl of Q5 Master Mix, adjust to pH 8.0, spin at 10,000 × g."
    for term in render.terms(text):
        assert term == term.lower()
        assert not re.search(r"[(){}\[\]<>]", term), term


FORMULAE = ["NaCl", "MgCl2", "dH2O", "NaOH", "H2O"]
ACRONYMS = ["DNA", "EDTA", "PCR", "PBS", "dNTP"]


@pytest.mark.parametrize("token", FORMULAE)
def test_a_formula_is_not_an_acronym(token):
    """NaCl is not spelled out. An all-capitals run is, whatever it spells."""
    kinds = {m.kind for m in render.detect(token)}
    assert "formula" in kinds
    assert "spell(" not in render.render(token)


@pytest.mark.parametrize("token", ACRONYMS)
def test_an_acronym_is_spelled(token):
    kinds = {m.kind for m in render.detect(token)}
    assert "acronym" in kinds


STEPS = [
    "Add 25 µl of Q5 High-Fidelity 2X Master Mix to the reaction tube.",
    "Add 0.5 µL of 10 mM dNTP mix to tube 3. Incubate at 98 °C for 30 seconds.",
    "Anneal at 55-65 °C for 20 s, then extend at 72 °C.",
    "Spin at 10,000 × g for 2 min at 4 °C.",
    "Dilute 1:10 in 1×PBS and adjust to pH 8.0 with 1 M NaOH.",
    "Add 2 U/µl Taq, then run the gel at 100 V for 45 min.",
    "Store at -20 °C. Catalogue M0492, 5' to 3'.",
    "Resuspend in dH2O containing MgCl2 and NaCl.",
    # From the second protocol, which is where the range and fraction cases
    # came from in the first place.
    "Take competent cells out of -80°C and thaw on ice for 20-30 mins.",
    "Heat shock the bottom 1/2 to 2/3 of the tube in a 42°C bath for 30-60 secs.",
    "Mix 1 - 5 µl of DNA (usually 10 pg - 100 ng) into 20-50 µL of cells.",
]


@pytest.mark.parametrize("text", STEPS)
def test_no_digit_survives_a_render(text):
    """A digit reaching Rime is a digit its own normaliser expands however it
    likes, which is the guesswork this module exists to remove."""
    assert not any(c.isdigit() for c in render.render(text)), render.render(text)


@pytest.mark.parametrize("text", STEPS)
def test_render_keeps_the_words_around_the_notation(text):
    """A match must not swallow the space in front of it.

    The first version anchored the optional sign before the number, the sign
    matched empty, and the whitespace was consumed with it, so "Add 25 µl"
    rendered as "Addtwenty five microlitres".
    """
    spoken = render.render(text)
    assert "  " not in spoken
    for word in ("Add", "Incubate", "Anneal", "Spin", "Dilute", "Store", "Resuspend"):
        if text.startswith(word):
            assert spoken.startswith(word + " "), spoken


@pytest.mark.parametrize("text", STEPS)
def test_matches_never_overlap_and_stay_in_order(text):
    cursor = 0
    for match in render.detect(text):
        assert match.start >= cursor, (match, text)
        assert text[match.start : match.end] == match.source
        cursor = match.end


def test_spoken_numbers_excludes_digits_that_belong_to_a_name():
    """The checker insists every quantity is spoken as words.

    It must not insist on the 2 in MgCl2 or the 0492 in M0492. Those are parts
    of a name, and demanding "two" appear inside "magnesium chloride" would
    reject a correct rendering.
    """
    expected = render.spoken_numbers("Add 25 µl of MgCl2, catalogue M0492.")
    assert expected == ("twenty five microlitres",)
