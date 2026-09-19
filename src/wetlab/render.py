"""Written text to speakable text.

Rime, in common with every speech engine, normalises written forms by guessing,
and laboratory notation is full of tokens it guesses wrong on. This module is
where the guessing is taken away from it.

Until 7 September it worked from spans a person had marked by hand, which meant
it only worked on a protocol somebody had already labelled. It now works from
plain text, which is what an uploaded protocol arrives as. See the detector at
the bottom of the file: that is the whole module now.

It is not the author of the spoken form. A language model writes that offline,
once per step, because it knows things no table will: that Taq is said "tack",
that MgCl2 is magnesium chloride, that M0492 is a catalogue number while 1 M is
molar. What is here checks that output by rule, supplies the index terms that
make a step findable, and renders a step when there is no prepared form.

Numbers are always spelled out. A digit that reaches Rime is a digit its own
normaliser expands however it likes, and 0.5 read as "point five" or "five" is
a different volume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNITS: dict[str, str] = {
    "µL": "microlitres",
    "uL": "microlitres",
    "μL": "microlitres",
    "mL": "millilitres",
    "L": "litres",
    "mM": "millimolar",
    "µM": "micromolar",
    "μM": "micromolar",
    "uM": "micromolar",
    "nM": "nanomolar",
    "M": "molar",
    "°C": "degrees celsius",
    "C": "degrees celsius",
    "s": "seconds",
    "sec": "seconds",
    # Plural abbreviations. Without "mins" in the table, "20-30 mins" did not
    # match the range pattern, fell through to two bare numbers, and the second
    # of them was read as negative thirty.
    "secs": "seconds",
    "min": "minutes",
    "mins": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "rpm": "r p m",
    "g": "times g",
    "pg": "picograms",
    "ng": "nanograms",
    "µg": "micrograms",
    "μg": "micrograms",
    "mg": "milligrams",
    "mol": "moles",
    "mmol": "millimoles",
    "µmol": "micromoles",
    "nmol": "nanomoles",
    "%": "percent",
    # Lowercase litre forms. These are what the corpus actually writes ("25 µl"),
    # and the span path never needed them because a human marked the span by
    # hand. Only the litre family is safe to add: lowercasing mM would collide
    # with mm, which is millimetres and not millimolar.
    "µl": "microlitres",
    "ul": "microlitres",
    "μl": "microlitres",
    "ml": "millilitres",
    # Added for the detector. Everything above was reachable only through a
    # hand-marked span, so the table only ever had to cover what the two
    # protocols in the corpus happened to use.
    "µg/mL": "micrograms per millilitre",
    "ng/µL": "nanograms per microlitre",
    "mg/mL": "milligrams per millilitre",
    "U/µL": "units per microlitre",
    "U/µl": "units per microlitre",
    "U": "units",
    "V": "volts",
    "°F": "degrees fahrenheit",
    "K": "kelvin",
    "kb": "kilobases",
    "bp": "base pairs",
    "nm": "nanometres",
    "mm": "millimetres",
    "cm": "centimetres",
    "kg": "kilograms",
    "µm": "micrometres",
    "ms": "milliseconds",
    "d": "days",
    # Units already written out as words. A protocol writes "30 seconds" at
    # least as often as "30 s", and with only the symbol in the table the
    # detector fell through to a bare number: the corpus's own time steps
    # produced no time term at all, so nothing could find them.
    "second": "seconds",
    "seconds": "seconds",
    "minute": "minutes",
    "minutes": "minutes",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "microlitre": "microlitres",
    "microlitres": "microlitres",
    "microliter": "microlitres",
    "microliters": "microlitres",
    "millilitre": "millilitres",
    "millilitres": "millilitres",
    "milliliter": "millilitres",
    "milliliters": "millilitres",
    "gram": "grams",
    "grams": "grams",
    "cycle": "cycles",
    "cycles": "cycles",
    "degrees Celsius": "degrees celsius",
    "degrees celsius": "degrees celsius",
}

#: The word that makes a quantity findable by what it *is* rather than by the
#: characters it is written with. This is the whole reason retrieval works on
#: this corpus: "temperature" appears in none of the cited protocol's step
#: texts, so a keyword search for it over raw text returns nothing. Whatever
#: recognises `98 °C` in order to speak it also supplies the term.
#: Words a person says when they mean a category, beyond the category's own
#: name. These exist because the questions are spoken: nobody at a bench asks
#: "what is the duration of the incubation", they ask "how long do I wait", and
#: neither "how" nor "wait" appears anywhere near a step about 30 seconds.
CATEGORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "volume": ("amount", "much"),
    "concentration": ("strength", "much"),
    "temperature": ("degrees", "hot", "cold", "warm"),
    "time": ("duration", "long", "wait"),
    "mass": ("weight", "amount", "much"),
    "speed": ("spin", "fast"),
    "force": ("spin", "fast"),
    "count": ("many",),
}

UNIT_CATEGORY: dict[str, str] = {
    "cycles": "count",
    "microlitres": "volume",
    "millilitres": "volume",
    "litres": "volume",
    "millimolar": "concentration",
    "micromolar": "concentration",
    "nanomolar": "concentration",
    "molar": "concentration",
    "percent": "concentration",
    "micrograms per millilitre": "concentration",
    "nanograms per microlitre": "concentration",
    "milligrams per millilitre": "concentration",
    "degrees celsius": "temperature",
    "degrees fahrenheit": "temperature",
    "kelvin": "temperature",
    "seconds": "time",
    "milliseconds": "time",
    "minutes": "time",
    "hours": "time",
    "days": "time",
    "picograms": "mass",
    "nanograms": "mass",
    "micrograms": "mass",
    "milligrams": "mass",
    "kilograms": "mass",
    "moles": "amount",
    "millimoles": "amount",
    "micromoles": "amount",
    "nanomoles": "amount",
    "r p m": "speed",
    "times g": "force",
    "units": "activity",
    "units per microlitre": "activity",
    "volts": "voltage",
    "kilobases": "length",
    "base pairs": "length",
    "nanometres": "length",
    "micrometres": "length",
    "millimetres": "length",
    "centimetres": "length",
}

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()

def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        out = f"{_ONES[hundreds]} hundred"
        return out + (f" {_int_to_words(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    out = f"{_int_to_words(thousands)} thousand"
    return out + (f" {_int_to_words(rest)}" if rest else "")


def spell_number(raw: str) -> str:
    """0.5 -> "zero point five", 98 -> "ninety eight".

    Decimals are spoken digit by digit after the point, which is how a pharmacy
    or a control tower says them, and which keeps 0.01 from collapsing toward
    0.1 in a listener's ear.
    """
    raw = raw.strip()
    if "." in raw:
        whole, frac = raw.split(".", 1)
        head = _int_to_words(int(whole)) if whole else "zero"
        tail = " ".join(_ONES[int(d)] for d in frac)
        return f"{head} point {tail}"
    return _int_to_words(int(raw))


def spell_unit(raw: str) -> str:
    """Write a unit out so it cannot be heard as its neighbour."""
    raw = raw.strip()
    if raw in UNITS:
        return UNITS[raw]
    lowered = {k.lower(): v for k, v in UNITS.items()}
    return lowered.get(raw.lower(), raw)


def _spell_letters(text: str) -> str:
    """spell() forces letter by letter, which is how EGTA stops sounding like EDTA."""
    return f"spell({text})"


def _spell_digits(text: str) -> str:
    """Any digit inside a name becomes a word. "Q5" -> "Q five", "tube 3" -> "tube three".

    A digit that reaches Rime is a digit its own normaliser expands however it
    likes, and that is the one thing mechanism 6.1 is for.

    The spelled number is separated from letters either side. Substituting in
    place turns "Q5" into "Qfive" and "2X" into "twoX", which Rime then reads as
    one invented word: a different wrong pronunciation, not a fix.
    """

    def spell(match: re.Match[str]) -> str:
        word = _int_to_words(int(match.group()))
        before = match.string[: match.start()]
        after = match.string[match.end() :]
        lead = " " if before[-1:].isalpha() else ""
        trail = " " if after[:1].isalpha() else ""
        return f"{lead}{word}{trail}"

    return re.sub(r"\d+", spell, text)


def _tidy(text: str) -> str:
    """Clean up the seams where a rewrite meets the text around it.

    A spoken form is substituted into a sentence without knowing what sits
    either side of it, which strands punctuation: a doubled comma where two
    rewrites meet, and a comma before a full stop where one ends a sentence.
    Rime speaks both. A doubled comma is a doubled pause, and ", ." is an
    audible stumble at the end of a sentence.
    """
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    # A comma immediately before a stronger stop is redundant; the stop wins.
    text = re.sub(r",\s*([.;:!?])", r"\1", text)
    # Nothing should open on a comma, and a trailing one leaves the utterance
    # hanging on a rising intonation that never resolves.
    text = re.sub(r"^\s*,\s*", "", text)
    text = re.sub(r"\s*,\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

#: Longest first, so mM is never matched as M and µL is never matched as L.
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

_NUM = r"[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?"
_SIGN = r"[-−–]?"
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

#: Kinds whose digits are quantities a listener has to act on, so the offline
#: pass must render them as words and the checker must insist on it. The kinds
#: left out carry digits that belong to a name: the 2 in MgCl2 and the 0492 in
#: M0492 are not amounts, and demanding "two" appear in "magnesium chloride"
#: would fail a correct rendering.
SPOKEN_NUMBER_KINDS = frozenset(
    {"quantity", "range", "ratio", "number", "multiplier", "ph", "rcf", "scientific", "prime"}
)

_SCAN = re.compile(
    "|".join(
        (
            # 1.5 × 10⁶ cells/mL. Before everything, because it contains a
            # multiplier, a bare number and an exponent that would each match.
            rf"(?P<scientific>(?P<sci_m>{_NUM})\s*[×x]\s*10\s*"
            rf"(?:\^\s*(?P<sci_e1>-?[0-9]+)|(?P<sci_e2>[⁰¹²³⁴⁵⁶⁷⁸⁹]+)))",
            # 10,000 × g. Relative centrifugal force, not a mass in grams.
            rf"(?P<rcf>(?P<rcf_n>{_NUM})\s*[×x]\s*g(?![A-Za-z]))",
            rf"(?P<ph>pH\s*(?P<ph_n>{_NUM}))",
            # 55-65 °C, before quantity, or the range reads as one temperature.
            rf"(?P<range>(?P<r_a>{_NUM})\s*(?:-|–|—|to)\s*"
            rf"(?P<r_b>{_NUM})\s*(?P<r_u>{_UNIT_ALT})(?![A-Za-z]))",
            rf"(?P<ratio>(?P<ra_a>{_NUM})\s*:\s*(?P<ra_b>{_NUM}))",
            # 1/2, 2/3. Before the unit-less range, or "1/2 to 2/3" reads as a
            # range from one to three with two slashes lying about in it.
            r"(?P<fraction>(?P<fr_n>[0-9]{1,2})\s*/\s*(?P<fr_d>[0-9]{1,2})(?![0-9]))",
            # A range with no unit after it: "20 to 30", "1 - 5". Kept after
            # the unit range so the unit is still picked up when there is one.
            rf"(?P<bare_range>(?P<br_a>{_NUM})\s*(?:-|–|—|to)\s*(?P<br_b>{_NUM})(?![0-9./]))",
            # The common case: 25 µl, 98 °C, 0.5 µM, -20 °C. No \s* between the
            # sign and the number: the sign can match empty, and the whitespace
            # would then be eaten from in front of the number, so the match
            # started at the space and "Add 25 µl" came back as "Addtwenty
            # five microlitres".
            rf"(?P<quantity>(?P<q_s>{_SIGN})(?P<q_n>{_NUM})\s*"
            rf"(?P<q_u>{_UNIT_ALT})(?![A-Za-z]))",
            # 2X Master Mix, 1×PBS. A letter may follow, so no guard against
            # one: 1×PBS is a strength and a reagent run together.
            rf"(?P<multiplier>(?P<x_n>{_NUM})\s*[×xX](?![a-z]))",
            r"(?P<prime>(?P<pr_n>[0-9])['′])",
            # One group for every name-shaped token, decided in the handler
            # rather than by which pattern is written first. As separate groups
            # the formula pattern matched DNA, the handler then rejected it as
            # not a formula, and the token was dropped instead of falling
            # through to the acronym rule.
            r"(?P<token>\b[A-Za-z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*\b)",
            rf"(?P<number>{_SIGN}{_NUM})",
        )
    )
)

_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs "
    "Ba La Ce Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
)


@dataclass(frozen=True)
class Match:
    """One thing found in written text, in both of the forms it is needed in."""

    source: str
    start: int
    end: int
    kind: str
    #: What the detector would say. A fallback and a reference, not the product:
    #: the offline pass writes the spoken form and this is what checks it.
    speech: str
    #: Plain lowercase words that should find this step in a keyword search.
    terms: tuple[str, ...]


def _plain_number(raw: str) -> str:
    return raw.replace(",", "")


def _spell_signed(raw: str) -> str:
    raw = _plain_number(raw.strip())
    if raw[:1] in "-−–":
        return f"minus {spell_number(raw[1:])}"
    return spell_number(raw)


def unit_spoken_forms(symbol: str) -> tuple[str, ...]:
    """Every way a unit might legitimately be written and still mean itself.

    Generous about spelling and strict about scale: microlitres, microliters
    and the symbol all mean the same volume, and millilitres does not. Used by
    the offline checker and by the round-trip scoring, which both need to
    decide whether a unit survived without rejecting a correct variant.
    """
    spoken = spell_unit(symbol)
    forms = {symbol.lower(), spoken}
    forms.add(spoken.replace("litre", "liter").replace("metre", "meter"))
    for word in spoken.split():
        if len(word) > 3:
            forms.add(word)
            forms.add(word.replace("litre", "liter").replace("metre", "meter"))
    # These are spelled plural, and a unit is written singular whenever the
    # quantity is one: "a ten centimetre plate". Without the singular that
    # scored as the unit having been lost, in a reading where it was said
    # correctly and heard correctly. Matching is on whole words, so adding the
    # singular cannot make the plural match something it should not.
    for form in list(forms):
        if form.endswith("s") and len(form) > 4:
            forms.add(form[:-1])
    return tuple(sorted(f for f in forms if f))


def unit_terms(symbol: str) -> tuple[str, ...]:
    """The category word, its spoken synonyms, and the unit's own words."""
    spoken = spell_unit(symbol)
    category = UNIT_CATEGORY.get(spoken)
    words = [w for w in spoken.replace("/", " ").split() if len(w) > 1]
    head: list[str] = []
    if category:
        head = [category, *CATEGORY_SYNONYMS.get(category, ())]
    return tuple(dict.fromkeys(head + words))


#: Denominators said as a word rather than as an ordinal number.
_DENOMINATOR = {2: "half", 3: "third", 4: "quarter"}


def _spoken_fraction(numerator: str, denominator: str) -> str:
    """1/2 is "one half", 2/3 is "two thirds", 5/8 is "five over eight".

    A protocol says to submerge the bottom half of a tube, not the bottom
    one-slash-two of it, and Rime reads a slash as the word "slash".
    """
    n, d = int(numerator), int(denominator)
    word = _DENOMINATOR.get(d)
    if word is None:
        return f"{_int_to_words(n)} over {_int_to_words(d)}"
    if n == 1:
        return f"one {word}"
    return f"{_int_to_words(n)} {word}s"


def _spell_each_digit(text: str) -> str:
    """Digits one at a time. A catalogue number is a label, not a count.

    M0492 is "M zero four nine two". Reading it as four hundred and ninety two
    loses the leading zero, and a listener writing it down gets a different
    part number from the one on the bottle.
    """
    return re.sub(
        r"[0-9]+",
        lambda m: " " + " ".join(_ONES[int(d)] for d in m.group()) + " ",
        text,
    )


def _looks_like_formula(token: str) -> bool:
    """NaCl and MgCl2 yes; DNA and PCR no.

    A formula is element-shaped pieces where at least one piece carries a
    lowercase letter or a digit. An all-capitals run with nothing else is an
    acronym, whatever its letters happen to spell.
    """
    pieces = re.findall(r"[A-Z][a-z]?[0-9]*", token)
    if len(pieces) < 2:
        return False
    symbols = [re.sub(r"[0-9]+", "", p) for p in pieces]
    if not any(p != p.upper() or any(c.isdigit() for c in p) for p in pieces):
        return False
    return sum(s in _ELEMENTS for s in symbols) >= len(symbols) - 1


def _classify_token(text: str) -> tuple[str, str, tuple[str, ...]]:
    """A name-shaped token, decided once: catalogue, formula or acronym."""
    if re.fullmatch(r"[A-Za-z]{1,3}[0-9]{3,}", text):
        return "catalogue", _spell_each_digit(text), ("catalogue", "part", text.lower())
    if _looks_like_formula(text):
        return "formula", _spell_digits(text), ("formula", "reagent", text.lower())
    letters = re.sub(r"[0-9]", "", text)
    # A single leading lowercase letter is a chemistry prefix, not a word:
    # the d in dNTP is deoxy and the whole token is said letter by letter.
    # Testing isupper() on the token itself leaves dNTP unspelled, which is
    # the one acronym this corpus actually contains.
    core = letters[1:] if letters[:1].islower() else letters
    if core.isupper() and len(core) >= 2:
        spoken = _spell_letters(letters)
        digits = re.sub(r"[^0-9]", "", text)
        if digits:
            spoken = f"{spoken} {_spell_each_digit(digits).strip()}"
        return "acronym", spoken, ("acronym", "reagent", text.lower())
    if re.fullmatch(r"[A-Z][0-9]+", text):
        return "acronym", _spell_digits(text), ("acronym", "reagent", text.lower())
    if text in UNITS or text.lower() in {u.lower() for u in UNITS}:
        # A unit with no number in front of it, as in "cells/mL". It is not a
        # reagent and indexing it as one would put every millilitre step in the
        # results for a reagent search.
        return "unit", text, unit_terms(text)
    # Mixed case with no element structure: Taq, HiFi. The detector has no
    # opinion on how these are said, which is exactly why a model writes the
    # spoken form and this only checks it.
    return "name", text, ("reagent", text.lower())


def _match_from(m: re.Match[str]) -> Match | None:
    kind = m.lastgroup
    text = m.group()
    for name in (
        "scientific",
        "rcf",
        "ph",
        "range",
        "ratio",
        "fraction",
        "bare_range",
        "quantity",
        "multiplier",
        "prime",
        "token",
        "number",
    ):
        if m.group(name) is not None:
            kind = name
            break
    else:
        return None

    def built(speech: str, terms: tuple[str, ...]) -> Match:
        return Match(
            source=text,
            start=m.start(),
            end=m.end(),
            kind=kind,
            speech=speech,
            terms=tuple(dict.fromkeys(t for t in terms if t)),
        )

    if kind == "scientific":
        mantissa = _spell_signed(m.group("sci_m"))
        exponent = m.group("sci_e1") or (m.group("sci_e2") or "").translate(_SUPER)
        return built(
            f"{mantissa} times ten to the power {_spell_signed(exponent)}",
            ("scientific", "power", _plain_number(m.group("sci_m"))),
        )
    if kind == "rcf":
        n = _plain_number(m.group("rcf_n"))
        return built(f"{_spell_signed(n)} times g", ("force", "speed", "centrifuge", n))
    if kind == "ph":
        n = _plain_number(m.group("ph_n"))
        return built(f"p H {_spell_signed(n)}", ("ph", "acidity", n))
    if kind == "range":
        a, b = _plain_number(m.group("r_a")), _plain_number(m.group("r_b"))
        unit = m.group("r_u")
        return built(
            f"{_spell_signed(a)} to {_spell_signed(b)} {spell_unit(unit)}",
            ("range", *unit_terms(unit), a, b),
        )
    if kind == "ratio":
        a, b = _plain_number(m.group("ra_a")), _plain_number(m.group("ra_b"))
        return built(
            f"{_spell_signed(a)} to {_spell_signed(b)}", ("ratio", "dilution", a, b)
        )
    if kind == "fraction":
        n, d = m.group("fr_n"), m.group("fr_d")
        return built(_spoken_fraction(n, d), ("fraction", "part", f"{n}/{d}"))
    if kind == "bare_range":
        a, b = _plain_number(m.group("br_a")), _plain_number(m.group("br_b"))
        return built(
            f"{_spell_signed(a)} to {_spell_signed(b)}", ("range", "between", a, b)
        )
    if kind == "quantity":
        n = _plain_number(m.group("q_s") + m.group("q_n"))
        unit = m.group("q_u")
        return built(f"{_spell_signed(n)} {spell_unit(unit)}", (*unit_terms(unit), n.lstrip("-−–")))
    if kind == "multiplier":
        n = _plain_number(m.group("x_n"))
        return built(f"{_spell_signed(n)} X", ("concentration", "strength", n))
    if kind == "prime":
        n = m.group("pr_n")
        return built(f"{_spell_signed(n)} prime", ("prime", "end", "oligonucleotide", n))
    if kind == "token":
        kind, speech, token_terms = _classify_token(text)
        if kind in ("name", "unit"):
            # Nothing to rewrite, so recording a match would only make the
            # renderer replace the word with itself. The term is still worth
            # having, so it is returned with the source unchanged.
            return Match(
                source=text,
                start=m.start(),
                end=m.end(),
                kind=kind,
                speech=text,
                terms=token_terms,
            )
        return built(speech, token_terms)
    return built(_spell_signed(text), ("number", _plain_number(text)))


def detect(text: str) -> tuple[Match, ...]:
    """Everything in `text` a speech engine would otherwise guess at.

    Matches never overlap and are returned in the order they appear.
    """
    found = [_match_from(m) for m in _SCAN.finditer(text)]
    return tuple(m for m in found if m is not None)


def render(text: str) -> str:
    """The detector's own spoken form of written text.

    This is the fallback and the reference, not the product. The spoken form
    that actually gets read aloud is written offline by a model and committed to
    spoken.yaml, because a table does not know how a word is said.
    """
    out: list[str] = []
    cursor = 0
    for match in detect(text):
        out.append(text[cursor : match.start])
        # A rewrite can end in a letter where the source ran straight on into
        # one, as 1×PBS does. Without this the two collide into "Xspell(PBS)"
        # and Rime reads the join as a single invented word.
        if out and out[-1][-1:].isalnum() and match.speech[:1].isalnum():
            out.append(" ")
        out.append(match.speech)
        cursor = match.end
        tail = text[cursor : cursor + 1]
        if tail.isalnum() and match.speech[-1:].isalnum():
            out.append(" ")
    out.append(text[cursor:])
    return _tidy("".join(out))


def terms(text: str) -> tuple[str, ...]:
    """The index terms for a step: every detected term, plus its own words."""
    collected: list[str] = []
    for match in detect(text):
        collected.extend(match.terms)
    return tuple(dict.fromkeys(t.lower() for t in collected))


def spoken_numbers(text: str) -> tuple[str, ...]:
    """The word forms the offline pass must produce for this text.

    Only quantities. A digit inside a name is excluded, because insisting on
    "two" inside "magnesium chloride" would reject a correct rendering.
    """
    out: list[str] = []
    for match in detect(text):
        if match.kind in SPOKEN_NUMBER_KINDS:
            out.append(match.speech)
    return tuple(out)
