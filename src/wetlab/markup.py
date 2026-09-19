"""Speech-engine markup, applied at the one place text becomes audio.

`spell(DNA)` is an instruction to Rime, not a word. It used to live in the
corpus text, which meant one string served two consumers with incompatible
needs: Rime, which requires the markup, and the language model, which must
never see it. The model was shown the marked-up text, repeated it in a reply,
and Rime was handed "spell Q" as words to say. The listener heard "spell".

So the markup is not in the text any more. It is applied in `tts_node`, the
single point where text becomes audio, to whatever is about to be synthesised.
Three things follow, and the third is the reason this is worth a module:

- The model never sees markup, so it cannot leak it.
- The markup cannot be corrupted in transit, because it is added after every
  component that could corrupt it.
- Anything the model composes itself is spelled correctly too. Under the old
  arrangement only text copied verbatim out of the corpus carried the markup,
  so a reagent the model named in its own sentence was read as a word.

Which terms are spelled is not a list kept here. It is read from the corpus:
the preparation pass already decided, per protocol, which terms a person reads
out letter by letter, and this uses that decision rather than a second one.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, Iterable

#: A term must not be bounded by letters to match. Without this, "LB" matched
#: inside "LB-agar" fine but also inside any longer word containing it.
#:
#: Digits are deliberately not part of that fence, and that is a fix rather than
#: an oversight. The boundary excluded them until 8 September, so the "Q" in
#: "Q5" was not a match: the digit after it was exactly the character the
#: lookahead rejected. "Q5" therefore reached Rime whole, and Rime's normaliser
#: reads a Q in front of a number as the sign for the Guatemalan quetzal, so the
#: protocol's own name was spoken as an amount of money. Every live run said it,
#: in the greeting, because the greeting is composed by the model from the raw
#: title rather than from the prepared corpus. See RIME_EVIDENCE.md section 1b.
#:
#: Nothing in the prepared corpus contains a digit, so widening this cannot
#: change what any authored step synthesises to. It only reaches text the model
#: wrote itself, which is the text this module exists to cover.
_BOUNDARY = r"(?<![A-Za-z]){}(?![A-Za-z])"


def spelled_terms(texts: Iterable[str]) -> tuple[str, ...]:
    """Every term the prepared corpus marks as read letter by letter.

    Longest first, so that a term containing another is wrapped whole rather
    than having its inner term wrapped first.
    """
    found: set[str] = set()
    for text in texts:
        found.update(re.findall(r"spell\(([^)]+)\)", text or ""))
    return tuple(sorted((t.strip() for t in found if t.strip()), key=len, reverse=True))


def apply(text: str, terms: Iterable[str]) -> str:
    """Wrap each known term in spell(), leaving anything already wrapped alone.

    Matching is case sensitive, plus the all-uppercase form. A protocol writes
    dNTP with a lowercase d and that capitalisation is the term; matching
    case-insensitively would also rewrite an ordinary word that happened to
    spell one, and "a" and "I" are terms in some protocols.
    """
    if not text:
        return text
    out = text
    for term in terms:
        variants = {term, term.upper()}
        for variant in sorted(variants, key=len, reverse=True):
            pattern = _BOUNDARY.format(re.escape(variant))
            # Skip an occurrence that is already inside spell(...).
            out = re.sub(pattern, lambda m, t=term: _wrap(m, t), out)
    return out


def _wrap(match: re.Match, term: str) -> str:
    """`spell(term)`, with a space where a digit was touching it.

    The markup already separates the two tokens, so "Q5" would come out as
    "spell(Q)5" and be read correctly. The space is for the same reason the
    prepared corpus writes "spell(Q) five" rather than "spell(Q)five": a closing
    bracket against a digit is a shape neither the corpus nor the measurements
    ever put in front of Rime, and there is no reason to be the first.
    """
    if _inside_spell(match):
        return match.group(0)
    text, start, end = match.string, match.start(), match.end()
    before = " " if start > 0 and text[start - 1].isdigit() else ""
    after = " " if end < len(text) and text[end].isdigit() else ""
    return f"{before}spell({term}){after}"


def _inside_spell(match: re.Match) -> bool:
    """True when this occurrence is already wrapped.

    Looks backwards for an unclosed `spell(` between the start of the string
    and the match. Cheap, and the alternative is wrapping a term twice, which
    Rime reads as the word "spell".
    """
    before = match.string[: match.start()]
    opened = before.rfind("spell(")
    if opened == -1:
        return False
    return ")" not in before[opened:]


#: A term can be split across two chunks of a stream, so a chunk is not safe to
#: transform until the word at its end is known to be complete. Everything up
#: to the last word boundary is emitted; the tail is held for the next chunk.
_TAIL = re.compile(r"[A-Za-z0-9]+$")


async def stream(chunks: AsyncIterable[str], terms: Iterable[str]) -> AsyncIterable[str]:
    """Apply the markup across a stream, without splitting a term in half.

    The model's text arrives in pieces that do not respect word boundaries, so
    "DNA" can arrive as "D" then "NA". Transforming each piece as it comes
    would miss exactly the terms this exists to catch.
    """
    terms = tuple(terms)
    held = ""
    async for chunk in chunks:
        buffer = held + chunk
        tail = _TAIL.search(buffer)
        if tail:
            # Hold back the trailing partial word.
            held, buffer = buffer[tail.start() :], buffer[: tail.start()]
        else:
            held = ""
        if buffer:
            yield apply(buffer, terms)
    if held:
        yield apply(held, terms)
