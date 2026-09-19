"""Did the information survive being spoken?

The round trip sends a step to Rime twice, once as the protocol writes it and
once as the offline pass prepares it, transcribes both with the same recogniser
the assistant uses, and asks of each numeric, unit and acronym token whether it
came back.

Scoring has to be generous about *form* and strict about *value*. A recogniser
may write twenty five or 25, microlitres or microliters, and all of those mean
the volume survived. It may not write twenty, because that is a different
volume, and the whole point of the exercise is that the difference between
those two is the difference between a working reaction and a wasted afternoon.

This is not a measure of how a person hears it. A recogniser is not an ear, and
a token can survive here while still being hard to follow under hood noise.
That limitation belongs next to the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import render

#: Kinds the detector emits that carry a value a listener has to act on.
NUMERIC = frozenset({"quantity", "range", "ratio", "number", "multiplier", "ph", "rcf", "scientific"})


@dataclass(frozen=True)
class Token:
    """One thing that has to survive, and how to tell whether it did."""

    source: str
    kind: str
    #: Surface forms any of which count as recovered.
    accepts: tuple[str, ...]


@dataclass(frozen=True)
class Score:
    token: Token
    recovered: bool


def _norm(text: str) -> str:
    """Lowercase, punctuation flattened, padded so word boundaries are testable.

    A digit running into a letter is split. A recogniser writes 2X and Q5 as
    single tokens, and without the split the 2 in 2X could not be found at all,
    so a perfectly recovered value scored as lost.
    """
    text = re.sub(r"(?<=[0-9])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[0-9])", " ", text)
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


#: Words after which spoken English optionally inserts "and": "two hundred and
#: fifty". Which side of the Atlantic you are on decides whether it is there,
#: and both are the same number.
_AND_AFTER = ("hundred", "thousand", "million", "billion")


def _with_and(spelled: str) -> set[str]:
    """The same number said with "and" wherever English allows one.

    `spell_number` writes "two hundred fifty". Rime says "two hundred and
    fifty" and the recogniser writes it down that way, so the form without the
    "and" matched nothing and a number that was said correctly and heard
    correctly scored as lost.
    """
    words = spelled.split()
    out = {spelled}
    for i, word in enumerate(words[:-1]):
        if word in _AND_AFTER and words[i + 1] != "and":
            out.add(" ".join(words[: i + 1] + ["and"] + words[i + 1 :]))
    return out


def _number_forms(value: str) -> set[str]:
    """A number as digits and as words, since either means it survived."""
    value = value.replace(",", "").lstrip("-−–")
    forms = {value}
    try:
        forms |= _with_and(render.spell_number(value))
    except (ValueError, IndexError, KeyError):
        pass
    if "." in value:
        # A recogniser may write "0.5" as "point five" or ".5".
        forms.add(value.lstrip("0"))
    return {f for f in forms if f}


def _unit_forms(symbol: str) -> set[str]:
    """Every way a recogniser might write a unit and still mean it.

    Shared with the offline checker, because both have to decide whether a
    unit survived and neither should reject a correct variant: Rime says
    microlitres, the recogniser may write microliters, and neither is a
    different quantity.
    """
    return set(render.unit_spoken_forms(symbol))


def tokens_in(text: str) -> tuple[Token, ...]:
    """Everything in a step that has to survive being spoken."""
    out: list[Token] = []
    for match in render.detect(text):
        if match.kind in NUMERIC:
            numbers = re.findall(r"[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?", match.source)
            for number in numbers:
                out.append(Token(number, "number", tuple(sorted(_number_forms(number)))))
            unit = re.sub(r"[-−–0-9,.\s:×x]", "", match.source)
            if unit and unit in render.UNITS:
                out.append(Token(unit, "unit", tuple(sorted(_unit_forms(unit)))))
        elif match.kind == "acronym":
            letters = re.sub(r"[^A-Za-z]", "", match.source).lower()
            digits = re.sub(r"[^0-9]", "", match.source)
            # A single letter is not evidence of anything. Q5 scored on "q"
            # alone matched almost any transcript, so the digit has to be part
            # of what is looked for.
            forms = {letters, " ".join(letters)}
            if digits:
                spelled = render.spell_number(digits)
                forms = {f"{f} {d}" for f in forms for d in (digits, spelled)}
            elif len(letters) < 2:
                continue
            out.append(Token(match.source, "acronym", tuple(sorted(forms))))
        elif match.kind in ("formula", "catalogue"):
            out.append(Token(match.source, match.kind, (match.source.lower(),)))
    return tuple(out)


#: Kinds where a one-character match proves nothing. A number may be a single
#: digit; a unit or a name may not.
_NEEDS_TWO = frozenset({"unit", "acronym", "formula", "catalogue"})


def _usable(form: str, kind: str) -> str | None:
    """A form as it will be compared, or None if it is too short to mean anything.

    Normalising drops non-ASCII, so the symbol µl collapses to l and °c to c.
    Left in, those matched any transcript containing a stray letter: one raw
    reading came back as "zero point five l of ten mm" with the unit plainly
    lost, and scored as recovered on both counts.
    """
    normalised = _norm(form).strip()
    if not normalised:
        return None
    if kind in _NEEDS_TWO and len(normalised.replace(" ", "")) < 2:
        return None
    return normalised


def score(text: str, transcript: str) -> tuple[Score, ...]:
    """For each token in `text`, whether `transcript` carries it."""
    haystack = _norm(transcript)
    out = []
    for token in tokens_in(text):
        forms = [f for f in (_usable(a, token.kind) for a in token.accepts) if f]
        out.append(Score(token, any(f" {f} " in haystack for f in forms)))
    return tuple(out)


def summarise(scores) -> dict:
    """Overall and by token type, which is how the result is reported."""
    scores = list(scores)
    out: dict = {}
    by_kind: dict[str, list[Score]] = {}
    for s in scores:
        by_kind.setdefault(s.token.kind, []).append(s)
    for kind, group in sorted(by_kind.items()):
        out[kind] = {
            "recovered": sum(s.recovered for s in group),
            "total": len(group),
        }
    out["all"] = {"recovered": sum(s.recovered for s in scores), "total": len(scores)}
    # Numeric tokens taken alone, because the claim is judged on those as well
    # as overall: a rendering that gets the words right and the numbers wrong
    # is worse than useless.
    numeric = [s for s in scores if s.token.kind == "number"]
    out["numeric_only"] = {
        "recovered": sum(s.recovered for s in numeric),
        "total": len(numeric),
    }
    return out


def percent(entry: dict) -> float | None:
    return round(100 * entry["recovered"] / entry["total"], 1) if entry["total"] else None
