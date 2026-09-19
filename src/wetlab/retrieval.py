"""Finding a step in a protocol too long to hold in context.

The model is given the current step, the five before it, and a table of
contents. Anything else it needs, it looks up. One tool, one round trip: the
same call takes either a question or a list of step numbers, because asking
"what temperature did you mention" and asking "read me step four again" are
the same operation from the model's side and splitting them would cost a
second call to find out which one it wanted.

Scoring is BM25 over the step text plus the terms the offline preparation pass
recorded. Those terms are what makes it work at all. Searching raw step text
for "temperature" returns nothing in the cited protocol, because the word
never appears in it; the steps say 98 °C. The detector supplies "temperature"
as a term for exactly that reason.

Embeddings were considered and are in docs/DEFERRED_OPTIONS.md. They would
handle a paraphrase this cannot, at the cost of a model call on every lookup
and an index that is no longer reviewable in a diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

#: How many steps come back. Five is what fits in a spoken answer without the
#: model having to summarise, and summarising retrieved steps is where a wrong
#: value would come from.
LIMIT = 5

# Letters and digits, with a dot only between them. Written as [a-z0-9.]+ it
# also swallowed the full stop at the end of a sentence, so "on ice." indexed
# as "ice." and a question about ice matched nothing.
_WORD = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")

#: Words too common to decide that a step is relevant. Candidacy here is
#: membership rather than rarity (see search), so without this list "what
#: colour is the bench" matches every step containing "the".
_STOPWORDS = frozenset(
    """a an and are as at be been by did do does for from had has have how i in
    is it its me my of on or say said should so than that the their then there
    these they this to was were what when where which who will with you your
    can could would about into over under again please tell show read""".split()
)


@dataclass(frozen=True)
class Entry:
    """One step, as the index sees it."""

    number: int
    step_id: str
    text: str
    terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    number: int
    step_id: str
    text: str
    #: Whether this step has been read aloud. The difference between what the
    #: listener has heard and what the document contains.
    read_aloud: bool


def tokenise(text: str) -> list[str]:
    """Lowercase word tokens, decimals kept whole and plurals folded.

    "0.5" has to survive as one token. Splitting on the point makes it two
    tokens, "0" and "5", which then match every other step with a zero or a
    five in it.

    A trailing s is dropped, so "primers" and "primer" are the same token. The
    step says primer and the scientist says primers, and without this the
    question "what did you say about the primers" matched nothing at all in a
    protocol with two primer steps in it. The rule is crude, and "celsius"
    becomes "celsiu", but it is applied to the query and the index alike so
    they still meet.
    """
    return [_fold(w) for w in _WORD.findall(text.lower())]


def _fold(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class Index:
    """BM25 over the corpus, built once at startup from committed data."""

    def __init__(self, entries: Sequence[Entry]) -> None:
        self._entries = tuple(entries)
        self._by_number = {e.number: e for e in self._entries}
        # Terms go through the tokeniser too. The offline pass writes some of
        # them as phrases ("master mix", "nuclease free water"), and left whole
        # those are single tokens no query word can ever equal.
        corpus = [
            tokenise(e.text) + [t for term in e.terms for t in tokenise(term)]
            for e in self._entries
        ]
        #: Which words each step can be found by, used to decide candidacy
        #: before BM25 decides order. See the note in search().
        self._words = [set(doc) for doc in corpus]
        # BM25Okapi divides by the average document length, so it cannot be
        # built from nothing.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def __len__(self) -> int:
        return len(self._entries)

    def search(
        self,
        *,
        query: str | None = None,
        steps: Sequence[int] | None = None,
        limit: int = LIMIT,
        read_aloud: Sequence[str] = (),
    ) -> tuple[Hit, ...]:
        """Steps matching a question, or the steps asked for by number.

        Passing both is not an error: the numbers are returned first and the
        query fills the rest, which is what "step four, and anything else about
        the primers" means.
        """
        heard = set(read_aloud)
        chosen: list[Entry] = []

        for number in steps or ():
            entry = self._by_number.get(number)
            if entry is not None and entry not in chosen:
                chosen.append(entry)

        if query and self._bm25 is not None:
            tokens = tokenise(query)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                asked = {t for t in tokens if t not in _STOPWORDS}
                # Candidacy is decided by sharing a word, and only the order
                # among candidates is decided by BM25.
                #
                # Ranking by score alone does not work here. BM25 weights a
                # word by how rare it is, and "temperature" is a term on seven
                # of the cited protocol's fourteen steps, which gives it an
                # inverse document frequency of exactly zero: every score comes
                # back 0.0 and a cutoff at zero returns nothing at all. But a
                # question about temperature does want all seven steps. The
                # words that identify a category are common in the index by
                # construction, so rarity is the wrong test for whether a step
                # is relevant, and the right one is whether the word is there.
                ranked = sorted(
                    (i for i in range(len(self._entries)) if self._words[i] & asked),
                    key=lambda i: (scores[i], -i),
                    reverse=True,
                )
                for i in ranked:
                    if len(chosen) >= limit:
                        break
                    if self._entries[i] not in chosen:
                        chosen.append(self._entries[i])

        return tuple(
            Hit(
                number=e.number,
                step_id=e.step_id,
                text=e.text,
                read_aloud=e.step_id in heard,
            )
            for e in chosen[:limit]
        )


def from_protocol(protocol) -> Index:
    """Build the index from a loaded protocol and its terms.

    Terms come through `terms_for`, which prefers the prepared form and falls
    back to the detector. Reading `protocol.spoken` directly instead left the
    index with no terms at all until the offline pass had been run, and an
    index with no terms cannot answer any question about a temperature or a
    duration, because those words are not in the step text.
    """
    entries = [
        Entry(
            number=step.index + 1,
            step_id=step.id,
            text=step.text,
            terms=tuple(protocol.terms_for(step)),
        )
        for step in protocol.steps
    ]
    return Index(entries)
