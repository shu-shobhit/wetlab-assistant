#!/usr/bin/env python3
"""Turn a page of protocol text into the YAML the assistant reads.

A protocol does not have to be written by hand. This takes the source document
and produces parsed.yaml: an ordered list of steps, each one action, with any
duration a person waits through declared as a timer.

The model is z-ai/glm-5.3, not the flash variant the preparation pass uses.
Splitting a document into steps is a judgement rather than a rewrite, it is
made once per protocol at about five cents, and the stronger model is worth its
nineteenfold price here in a way it is not worth per step.

The prompt carries four rules because each maps to a way this goes wrong. The
last one matters more than it looks: the cited NEB protocol is two reference
tables and prose with no numbered steps at all, so turning it into an ordered
sequence is authoring rather than parsing, and the output is committed so the
choice is visible in the repository.

There is no review gate. The result is written and loaded, and the CITATION
file beside it says where the text came from.

    python scripts/ingest_protocol.py addgene_transformation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wetlab import render  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "z-ai/glm-5.3"

SYSTEM = """\
You convert a laboratory protocol document into an ordered list of steps to be
read aloud, one at a time, to a scientist whose hands are busy.

Return JSON only:
{"title": "...", "steps": [{"text": "...", "timers": [{"label": "...", "seconds": 0}]}]}

Six rules. Each one is here because it is a way this goes wrong.

1. ONE STEP IS ONE ACTION A PERSON PERFORMS. Two reagents added separately are
   two steps, because they are pipetted separately. One sentence describing two
   things a person does is two steps.
       "mix and incubate on ice"  ->  two steps

   A block that repeats is one step per stage inside it, not one step for the
   block. Collapsing a repeated block gives a single step carrying every value
   in it, read out in one breath to somebody who cannot see the text.
       "25-35 cycles of: 98°C 5-10 s, 50-72°C 10-30 s, 72°C 20-30 s/kb"
       ->  three steps, each saying it is part of the 25-35 cycles

2. EVERY VALUE APPEARS VERBATIM. No rounding, no paraphrase, no unit changes,
   no narrowing a range to a single number. This is the rule that matters most:
   a changed value is read aloud as fact to somebody holding a pipette.
       "20-30 mins" stays "20-30 mins", never "30 mins"

3. A VALUE THE SOURCE GIVES AS A RULE STAYS A RULE. Protocols often say how to
   work a number out rather than giving one. Write the rule, in the source's
   words. Never compute a number from it and never supply a typical one: the
   listener cannot tell an invented number from a measured one.
       "3°C above the Tm of the lower Tm primer"  stays as written
       "30 seconds per kilobase"                  stays as written

   Dropping the rule is as wrong as replacing it. A range whose selection rule
   has been left off asks the person to pick a value with nothing to pick it
   by. Carry the whole clause.

4. A DURATION A PERSON WAITS THROUGH BECOMES A DECLARED TIMER. A duration a
   machine runs by itself does not, and neither does one with no definite
   length. For a range, use the shorter value and leave the range in the step
   text. The label is announced aloud, so write it as words with no digits.
       "on ice for 20-30 mins"  ->  timer, 1200 seconds,
                                    "twenty to thirty minute incubation on ice"
       "incubate overnight"     ->  no timer, no definite length

5. EACH STEP IS HEARD ALONE. It is read on its own, with nothing on screen and
   no way to look back. So it has to stand by itself: name what is being acted
   on rather than referring back to it, and if a step only means something
   inside a repeat or a condition, say so in the step.
       "denature at 98°C for 10 seconds"  read alone does not say it happens
       twenty five to thirty five times

6. NOTHING IS INVENTED AND NOTHING IS DROPPED. If the source is ambiguous, the
   step says what the source says, ambiguity included. Do not add safety
   advice, explanations, or steps the document does not contain.

7. IF THE DOCUMENT DESCRIBES MORE THAN ONE VARIANT, USE ONLY ONE. A document
   that lays out alternatives side by side is several protocols printed at
   once, and mixing values across them produces one that is none of them. Take
   every value from the variant the request names. If it names none, take the
   first the document gives and say which in the title.

Ignore anything that is not a step the person performs: equipment lists,
reagent lists, introductions, pro-tips, commentary, FAQs. A pro-tip is advice
about a step, not a step.
"""


def user_prompt(source: str, variant: str = "") -> str:
    """The document, and which of its variants to take.

    NEB's page is a component table with a 25 µl column and a 50 µl column: two
    protocols printed side by side, and the model picked one on its own. Which
    reaction the corpus describes is not a thing to leave to a sampling
    temperature, so it is named here and recorded in the file header.
    """
    head = f"Protocol document:\n\n{source.strip()}"
    if variant.strip():
        return f"{head}\n\nUse only this variant of the protocol: {variant.strip()}"
    return head


#: A whole document at once, with a reasoning model, takes minutes. The first
#: value here was 300 seconds and it was actively harmful rather than merely
#: short: the timeout cut off a generation that was going well, and the retry
#: came back with a different and worse answer, collapsing three thermocycling
#: stages into one thirty-word step and dropping the clause that says how to
#: choose the annealing temperature. A slow answer is worth waiting for.
REQUEST_TIMEOUT_S = 900


async def ingest(source: str, key: str, variant: str = "", *, retries: int = 2) -> dict:
    """One document to steps, with retries.

    The first version had none, and a run came back with `content` set to null:
    the provider answered, the response was well formed, and the field the
    output goes in was empty. `json.loads(None)` then raised a TypeError from
    inside the json module, so a transient provider hiccup surfaced as a stack
    trace about types and the whole ingestion had to be started again by hand.
    """
    body = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(source, variant)},
        ],
    }
    last = "no attempt made"
    for attempt in range(retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
            content = payload["choices"][0]["message"]["content"]
            if not content:
                raise ValueError("the model returned an empty response")
            parsed = json.loads(content)
            usage = payload.get("usage") or {}
            print(f"  {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out")
            return parsed
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            print(f"  attempt {attempt + 1} failed: {last}", file=sys.stderr)
            await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"ingestion failed after {retries + 1} attempts: {last}")


def check(source: str, parsed: dict) -> list[str]:
    """What can be checked by rule, which is less than for the spoken form.

    Whether the steps are the right steps is a judgement, and the source sits
    next to the output so it can be read against it. What a rule can catch is a
    value that appears in a step but not in the document it came from.
    """
    problems: list[str] = []
    steps = parsed.get("steps") or []
    if not steps:
        return ["no steps"]

    # Thousands separators are dropped from the source as well as from the
    # step. The source writes "< 1,000 ng" and the step writes "1000 ng", which
    # is the same quantity; comparing them literally reported the step as
    # inventing a number that is sitting in the document.
    haystack = re.sub(r"(?<=[0-9]),(?=[0-9])", "", re.sub(r"\s+", " ", source.lower()))
    for i, step in enumerate(steps, start=1):
        text = str(step.get("text", "")).strip()
        if not text:
            problems.append(f"step {i}: empty")
            continue
        for match in render.detect(text):
            if match.kind not in render.SPOKEN_NUMBER_KINDS:
                continue
            # The value has to be somewhere in the source. Digits only: the
            # model may reasonably write "20 to 30 minutes" where the source
            # wrote "20-30 mins", and that is the same quantity.
            #
            # A thousands separator is part of the number rather than a
            # boundary. Without the comma group, "250-1,000" split into 250, 1
            # and 000, and then looked for "000" in a document that says
            # "1,000": a false report on one side and, worse, a lone "1" that
            # matches almost any document on the other.
            for number in re.findall(r"[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?", match.source):
                if number.replace(",", "") not in haystack:
                    problems.append(f"step {i}: {number!r} is not in the source document")
        for timer in step.get("timers") or []:
            label = str(timer.get("label", ""))
            if any(c.isdigit() for c in label):
                problems.append(f"step {i}: timer label has a digit in it: {label!r}")
            if float(timer.get("seconds", 0)) <= 0:
                problems.append(f"step {i}: timer has no length")
    return problems


def to_yaml(protocol_id: str, parsed: dict, variant: str = "") -> str:
    steps = []
    for i, step in enumerate(parsed.get("steps") or [], start=1):
        entry: dict = {"id": f"s{i}", "text": str(step["text"]).strip()}
        timers = []
        for j, timer in enumerate(step.get("timers") or [], start=1):
            timers.append(
                {
                    "id": f"s{i}.timer{j}",
                    "label": str(timer["label"]).strip(),
                    "seconds": float(timer["seconds"]),
                }
            )
        entry["timers"] = timers
        steps.append(entry)
    doc = {"id": protocol_id, "title": str(parsed.get("title", "")).strip(), "steps": steps}
    header = (
        f"# Ingested from source.md by scripts/ingest_protocol.py using {MODEL}.\n"
        "#\n"
        "# Splitting a document into steps is an authoring decision, not a parse.\n"
        "# The source sits beside this file so the two can be read against each\n"
        "# other, and CITATION says where the source came from.\n"
    )
    if variant.strip():
        header += f"#\n# Variant requested: {variant.strip()}\n"
    return header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


async def run(protocol_id: str, corpus: Path, dry_run: bool, variant: str = "") -> int:
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    root = corpus / protocol_id
    source_path = root / "source.md"
    if not source_path.exists():
        print(f"no source at {source_path}", file=sys.stderr)
        return 2
    source = source_path.read_text()

    print(f"ingesting {protocol_id} with {MODEL}" + (f", variant: {variant}" if variant else ""))
    parsed = await ingest(source, key, variant)
    problems = check(source, parsed)

    steps = parsed.get("steps") or []
    print(f"  {len(steps)} steps, {sum(len(s.get('timers') or []) for s in steps)} timers")
    for problem in problems:
        print(f"  PROBLEM {problem}")

    text = to_yaml(protocol_id, parsed, variant)
    if dry_run:
        print("\n" + text)
        return 1 if problems else 0

    out = root / "parsed.yaml"
    out.write_text(text)
    print(f"\nwritten to {out}")
    if problems:
        print("read the problems above against source.md before trusting this")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("protocol")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--dry-run", action="store_true", help="print the YAML, write nothing")
    ap.add_argument(
        "--variant",
        default="",
        help="which variant to take when the document describes several, "
        'e.g. "the 50 microlitre reaction column"',
    )
    args = ap.parse_args()
    return asyncio.run(run(args.protocol, args.corpus, args.dry_run, args.variant))


if __name__ == "__main__":
    raise SystemExit(main())
