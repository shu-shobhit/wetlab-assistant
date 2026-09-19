"""Checking what a model wrote before it is committed and spoken as fact.

The offline pass asks a language model for the spoken form of each step. That
is the right tool for the job: saying how a word is pronounced needs knowledge
no table has. It is also a tool that can quietly change a number, and a changed
number here is committed to the repository and then read aloud with complete
confidence to somebody holding a pipette. So something has to check it, and
that something has to be independent of the thing it is checking, or it is just
the model agreeing with itself.

It used to be a rule table built from the detector: every quantity the detector
found had to appear in the output in the wording the detector expected. Over
thirty prepared steps that produced five false rejections and caught nothing.
Every time it fired, the model was right and the table was missing an entry:
"mins" was not a unit, so 20-30 mins parsed as negative thirty; a fraction was
two numbers with a slash between them; picograms were unknown. A fixed table
does not generalise to notation nobody thought of, which is most notation.

What replaces it is a second model, from a different family than the one that
wrote the text, asked only whether every value survived. Independence is kept,
the table is gone, and notation nobody tabulated is covered.

Two rules stay, because neither needs a table and neither ever misfired: no
digit may survive, and the index terms must be plain lowercase words. What is
still not checked mechanically is meaning: whether the step says what the
source said is what review by eye is for, and what the round trip measures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

#: The verifier. A different family from the writer on purpose: two models that
#: share training and architecture share failure modes, and a checker that
#: fails in the same places as the thing it checks is not a checker.
VERIFIER_MODEL = "openai/gpt-5-mini"

#: An index term list feeds a keyword search. Too few and a step cannot be
#: found; too many and BM25's scoring is diluted by words that describe every
#: step equally. The live pass produced lists from one entry to nineteen.
MIN_TERMS = 3
MAX_TERMS = 12

_WORDY = re.compile(r"^[a-z][a-z0-9.\- ]*$")

VERIFY_SYSTEM = """\
You check whether a laboratory protocol step, rewritten to be read aloud, still
says exactly what the original said. A scientist will hear the rewrite while
holding a pipette and cannot see the original.

You are not asked whether it reads well. You are asked whether anything a
person has to act on changed.

The rewrite is allowed to change how something is written. It is not allowed to
change what is there. Judge every difference by that line.

Report a problem when, and only when:
- a quantity changed, was dropped, or was added.
      25 µl -> "twenty"                      the value changed
- a unit changed scale. A thousandfold error looks like a spelling difference
  and is not one.
      µl -> "millilitres"                    a thousand times wrong
- a named thing changed, vanished, or appeared: a reagent, a sample, a piece of
  equipment, a place. Naming something the original left unnamed is adding it,
  even when the name is obviously the right one.
      "out of minus eighty degrees celsius" -> "out of the freezer"
- an instruction was gained or lost. Both directions matter. A dropped action
  is one the listener will never know was there.
      "mix, then spin" -> "mix"              the spin is gone
- a qualifying clause was dropped: a condition, a limit, or the rule that says
  how to choose a value. A range with its selection rule removed leaves the
  person to guess.
      "50-72°C, three above the primer Tm" -> "fifty to seventy two degrees"
- spell() is wrapped around something a person would say as a word rather than
  as letters. Spelling an ordinary word makes it unrecognisable.
      spell(GENTLY), spell(Taq)              said as words, not letters
  A symbol read out as letters belongs in spell(), so spell(Tm) is correct.

Do NOT report a problem when:
- notation was written the way it is said. That is the entire point.
      25 -> "twenty five", µl -> "microlitres", MgCl2 -> "magnesium chloride",
      DNA -> spell(DNA)
- a spelling or regional variant differs while the value does not.
      "microlitres" and "microliters" are the same volume
- punctuation, phrasing, sentence splitting or word order changed while the
  meaning did not.
      brackets becoming commas, a slash becoming "and", one long sentence
      becoming two
- emphasis was lost. Speech cannot shout, and saying it plainly is correct.
      GENTLY -> "gently"

Return JSON only:
{"ok": true}
or
{"ok": false, "problems": ["one short sentence per problem"]}
"""


@dataclass
class Result:
    step_id: str
    speech: str
    terms: tuple[str, ...]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check(step_id: str, source: str, speech: str, terms) -> Result:
    """The rules that need no table. Cheap, local, and never wrong so far."""
    terms = tuple(terms or ())
    result = Result(step_id=step_id, speech=speech, terms=terms)
    problems = result.problems

    if not speech.strip():
        problems.append("empty spoken form")
        return result

    # A digit reaching Rime is a digit its own normaliser expands however it
    # likes, which is the guessing this whole pass exists to remove.
    digits = sorted({c for c in speech if c.isdigit()})
    if digits:
        problems.append(f"digits left in the spoken form: {''.join(digits)}")

    if not MIN_TERMS <= len(terms) <= MAX_TERMS:
        problems.append(f"{len(terms)} terms, wanted {MIN_TERMS} to {MAX_TERMS}")
    for term in terms:
        if not _WORDY.match(term):
            problems.append(f"term {term!r} is not a plain lowercase word")

    return result


def verify_prompt(source: str, speech: str) -> str:
    return f"Original step:\n{source.strip()}\n\nRewritten to be read aloud:\n{speech.strip()}"


def read_verdict(content: str) -> list[str]:
    """Turn the verifier's answer into a list of problems.

    A verifier that cannot be parsed is treated as a problem rather than as a
    pass. Silence is not approval when the thing being approved gets spoken to
    somebody who cannot check it.
    """
    try:
        verdict = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return [f"verifier returned something that is not JSON: {str(content)[:120]}"]
    if not isinstance(verdict, dict):
        return [f"verifier returned {type(verdict).__name__}, wanted an object"]
    if verdict.get("ok") is True:
        return []
    problems = verdict.get("problems")
    if isinstance(problems, str):
        problems = [problems]
    if not problems:
        return ["verifier said not ok but gave no reason"]
    return [f"verifier: {str(p).strip()}" for p in problems if str(p).strip()]
