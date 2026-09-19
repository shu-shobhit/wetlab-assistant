#!/usr/bin/env python3
"""Which words Rime does not know actually reach Rime.

`scripts/build_lexicon.py` asks `POST /oov` about every word in the corpus
*sources*. That is the right question for building a lexicon and the wrong one
for judging risk, because the product never speaks the source. It speaks the
prepared form, where a unit symbol has been written out as words and anything
read letter by letter sits inside spell().

So a word can be out of Rime's vocabulary and never be sent to it. This splits
the list three ways:

  spoken as a word     Rime predicts a pronunciation, and may be wrong
  inside spell()       read letter by letter, so nothing is predicted
  never sent           the preparation pass rewrote it before synthesis

Only the first group is a hazard. The other two are the offline pass removing
a problem as a side effect of doing something else.

    python scripts/oov_reach.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml


def spoken_vocabulary(corpus: Path, protocols) -> tuple[set[str], set[str]]:
    """Every word the prepared forms send to Rime, split by how it is sent."""
    plain_words: set[str] = set()
    spelled: set[str] = set()
    for name in protocols:
        path = corpus / name / "spoken.yaml"
        if not path.exists():
            continue
        for entry in (yaml.safe_load(path.read_text()) or {}).values():
            text = str(entry.get("speech", ""))
            for inner in re.findall(r"spell\(([^)]*)\)", text):
                spelled.add(inner.lower())
            # What is left after removing the spell() calls is what Rime is
            # asked to pronounce.
            outside = re.sub(r"spell\([^)]*\)", " ", text)
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", outside):
                plain_words.add(word.lower())
    return plain_words, spelled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--oov", type=Path, default=Path("data/oov.json"))
    ap.add_argument("--out", type=Path, default=Path("data/oov_reach.json"))
    args = ap.parse_args()

    if not args.oov.exists():
        print(f"no {args.oov}; run scripts/build_lexicon.py first", file=sys.stderr)
        return 2
    record = json.loads(args.oov.read_text())
    protocols = record.get("protocols") or []
    oov = [str(w).lower() for w in record.get("out_of_vocabulary") or []]

    plain, spelled = spoken_vocabulary(args.corpus, protocols)
    groups: dict[str, list[str]] = {"spoken_as_a_word": [], "inside_spell": [], "never_sent": []}
    for word in oov:
        if word in plain:
            groups["spoken_as_a_word"].append(word)
        elif word in spelled:
            groups["inside_spell"].append(word)
        else:
            groups["never_sent"].append(word)

    out = {
        "source": str(args.oov),
        "protocols": protocols,
        "out_of_vocabulary": len(oov),
        "groups": {k: sorted(v) for k, v in groups.items()},
        "note": (
            "Only spoken_as_a_word is a live risk: Rime predicts a pronunciation "
            "for those and may be wrong. Words inside spell() are read letter by "
            "letter, and words in never_sent were rewritten by the preparation "
            "pass and are not sent to Rime at all."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(oov)} words out of Rime's vocabulary across {len(protocols)} protocols\n")
    labels = {
        "spoken_as_a_word": "spoken as a word, so Rime predicts it",
        "inside_spell": "inside spell(), read letter by letter",
        "never_sent": "never sent, rewritten by the preparation pass",
    }
    for key, label in labels.items():
        words = sorted(groups[key])
        print(f"{len(words):3d}  {label}")
        if words:
            print(f"     {', '.join(words)}")
    removed = len(groups["inside_spell"]) + len(groups["never_sent"])
    print(f"\nthe offline pass removes {removed} of {len(oov)} before synthesis")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
