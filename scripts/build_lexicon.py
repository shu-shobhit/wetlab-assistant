#!/usr/bin/env python3
"""Ask Rime which words in the corpus it does not know.

Rime's `/oov` endpoint takes text and returns the words outside its lexicon.
Those are the words it will guess at, and a reagent name it guesses at is the
failure hazard class H4 exists for.

**Scope, decided 6 Sep 2026.** This runs `/oov` and stops. The fix for an
out-of-vocabulary word is `/phonemize`, which needs a recording of a person
saying it, and those recordings are not being made for this build. So what ships
is the finding, not the fix: `data/oov.json` says exactly which words Rime does
not know, which is the input to the recording step whenever it happens, and is a
real result on its own. Until then H4 spans fall back to spelling their digits
out, which is better than nothing and worse than a phoneme string.

    python scripts/build_lexicon.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wetlab import protocol as protocol_mod  # noqa: E402
from wetlab import render  # noqa: E402

OOV_URL = "https://users.rime.ai/oov"
OUT = Path("data/oov.json")

#: Every protocol whose words matter, which is every protocol in the corpus.
#: The second cited protocol was left out when it was added, so the record said
#: seventy one words had been checked while fourteen steps of new vocabulary,
#: including every word in the transformation protocol, had not been asked
#: about at all.
PROTOCOLS = ("handwritten", "neb_q5_m0492", "addgene_transformation")


def candidate_words(corpus: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Reagent names and every other word the assistant will actually say.

    Drawn from the corpus rather than a hand-written list, so a protocol added
    later brings its own vocabulary with it.
    """
    reagents: set[str] = set()
    everything: set[str] = set()
    sources: dict[str, list[str]] = {}

    for name in PROTOCOLS:
        # Only the source words are needed here, so an unprepared protocol is
        # still worth asking Rime about.
        proto = protocol_mod.load(corpus, name, require_spoken=False)
        for step in proto.steps:
            for word in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", step.text):
                everything.add(word)
                # Rime splits a hyphenated word before checking it, so it returns
                # "nuclease" for "nuclease-free". Index the parts as well, or the
                # record says the word appears nowhere.
                for part in {word, *word.split("-")}:
                    if part:
                        sources.setdefault(part.lower(), []).append(f"{name}/{step.id}")
            # Reagent names used to come from spans a person had marked H3 or
            # H4 by hand. The detector finds the same tokens without one.
            for match in render.detect(step.text):
                if match.kind in {"acronym", "formula", "name"}:
                    reagents.add(match.source)
    return sorted(reagents | everything), sources


def ask_rime(key: str, words: list[str]) -> list[str]:
    """`/oov` takes text and returns the words it does not know."""
    response = requests.post(
        OOV_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"text": " ".join(words)},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        for field in ("oov", "words", "out_of_vocabulary", "result"):
            if field in payload:
                return list(payload[field])
        return [payload]
    return list(payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("RIME_API_KEY")
    if not key:
        print("RIME_API_KEY is not set", file=sys.stderr)
        return 2

    words, sources = candidate_words(args.corpus)
    print(f"asking Rime about {len(words)} distinct words from {len(PROTOCOLS)} protocols")
    unknown = ask_rime(key, words)

    record = {
        "endpoint": OOV_URL,
        "protocols": list(PROTOCOLS),
        "words_checked": len(words),
        "out_of_vocabulary": unknown,
        "where_each_appears": {
            str(w).lower(): sorted(set(sources.get(str(w).lower(), []))) for w in unknown
        },
        "note": (
            "The fix for each of these is a phoneme string from /phonemize, which "
            "needs a recording of a person saying the word. Those recordings were "
            "not made for this build, so this file is the finding rather than the "
            "fix. H4 spans currently fall back to spelling their digits out."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(unknown)} out of vocabulary:")
    for word in unknown:
        where = record["where_each_appears"].get(str(word).lower(), [])
        print(f"  {word}   {', '.join(where) if where else '(reagent span)'}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
