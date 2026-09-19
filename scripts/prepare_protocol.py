#!/usr/bin/env python3
"""Write the spoken form of every step, once, offline.

A language model does this because it knows things no rule table will: that Taq
is said "tack" and not spelled, that MgCl2 is magnesium chloride, that M0492 is
a catalogue number while 1 M is molar. None of it happens at run time, so it
costs nothing in response time and cannot fail during a demo.

One model proposes and a second, from a different family, accepts. A corrupted
value here is committed and then read aloud as fact to somebody holding a
pipette, so something has to check it, and that something has to be independent
of the thing it is checking or it is just the model agreeing with itself.

Two local rules run first, because neither needs a table and both are free: no
digit may survive, and the terms must be plain lowercase words. Anything
failing either, or failing the verifier, is reported with its step number and
not written.

One request per step, so a bad output is isolated to one step and can be re-run
alone with --only.

    python scripts/prepare_protocol.py neb_q5_m0492
    python scripts/prepare_protocol.py neb_q5_m0492 --only s3 s9
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wetlab import prepare, protocol as protocol_mod  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
BATCH_URL = "https://openrouter.ai/api/beta/batches"

#: Tested on 7 September against the hard cases: ratios, ranges, a negative
#: temperature, units outside any table, and the spelled-versus-read acronym
#: split. It got all of them right and left no digit in any output. Sonnet was
#: dropped because it costs seven times as much and this needs knowledge rather
#: than reasoning.
MODEL = "z-ai/glm-5.3-flash"

#: Enough to finish a hundred steps in about two minutes without tripping a
#: rate limit.
CONCURRENCY = 8

SYSTEM = """\
You prepare laboratory protocol steps to be read aloud by a speech engine.

Return JSON only, with exactly two keys:
  "speech": a string, the step written so it is said correctly when read aloud
  "terms":  a JSON array of 4 to 10 strings, each a plain lowercase word a
            person might search this step by. An array, not one string.

Rules for "speech", all of which matter because the listener is holding a
pipette and cannot see the text:

- Keep every value exactly as it is. Never round, never convert a unit, never
  drop a quantity. This is the one thing that must not change.
- Write every number as words: "twenty five", not "25". A decimal is said digit
  by digit after the point: 0.5 is "zero point five", 0.01 is "zero point zero
  one".
- Write every unit out in full as it is said: "microlitres", "degrees celsius",
  "millimolar", "times g", "units per microlitre".
- spell() marks anything a person reads out letter by letter, and it is the only
  way to mark it. Never space letters apart by hand: "D N A" is the same
  intention written in a form the speech engine does not recognise, and the two
  spellings of the same term in one protocol will not sound alike.
- Whether a thing is spelled is decided by what a person saying this step aloud
  would do, not by how the text is capitalised. Capitals often mean emphasis, or
  are simply how a term is written, and neither makes a thing spelled.
      spell(DNA), spell(EDTA), spell(dNTP)   letters, so spelled
      spell(Tm)                              a symbol, still said as letters
      Taq -> "tack"                          a word, said as a word
      GENTLY -> "gently"                     capitals for emphasis
- Write anything else that is not said the way it is written as it is said
  instead.
      MgCl2 -> "magnesium chloride"          a formula, said as its name
      dH2O -> "distilled water"
      M0492 -> "M zero four nine two"        an identifier, read out
- Never leave a digit anywhere in "speech".
- Keep a quantity next to the thing it measures, in the order the step puts
  them. A listener who has heard the noun has stopped waiting for a number.
      "a ten centimetre plate", not "a plate, ten centimetres"
- Say what the step says and nothing else. Do not explain, warn, advise, or add
  a word that only makes the step easier to picture. Adding something true is
  still adding it, and the listener cannot tell which words came from the
  protocol.
      "out of minus eighty degrees celsius" does not become
      "out of the minus eighty degrees celsius freezer"
- No markup of any kind. spell() is the only thing in the output that is not
  plain speech. Everything else is read aloud literally, brackets included.
- Keep sentences under twenty five words, and under fifteen where you can. A
  long sentence arrives without a breath in it. Split a long step into two
  sentences rather than leaving anything out of it. Splitting is not licence to
  add: build the new sentences out of the words the step already uses.

Rules for "terms". These feed a keyword search, and the questions arrive
spoken, so the terms have to be the words a person says out loud:

- Plain lowercase words only. No markup, no spell(), no punctuation.
- Include what each quantity *is*, not just what it says. A step with 98 °C in
  it carries "temperature"; a step with 30 seconds carries "time". Neither
  word appears anywhere in a protocol, and they are what somebody asks by.
- Include every different word a person might use for the same thing, not just
  the name of the category. Someone asking about a duration says "how long do
  I wait", "what was the duration" or "how much time", so a timed step carries
  time, duration, long and wait. A volume carries volume, amount and much. A
  temperature carries temperature, hot and degrees. A count carries many.
  Two or three alternatives per quantity, because you cannot know which word
  the question will use.
- Include the reagents, the equipment and the action.
"""


def user_prompt(step) -> str:
    """Just the step.

    A rule table's reading of the notation used to be appended here as a
    starting point. It was dropped along with the table: on the notation the
    table did not know it was confidently wrong, offering "negative thirty" for
    20-30 mins, and handing a model a wrong answer to anchor on is worse than
    handing it nothing.
    """
    return f"Step: {step.text}"


def as_terms(value) -> list[str]:
    """Terms as a list, however the model chose to write them.

    Two shapes have turned up in live runs, both usable and neither an array.
    Some steps came back with `terms` as one comma-separated string, and
    iterating that yields single characters, so the rules saw forty-eight terms
    of which several were a space. Others came back space-separated with no
    commas at all, which splitting on commas turns into a single term.

    The rules were right to reject both. The output was fine, so this is a
    parsing job rather than a rejection.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if not isinstance(value, str):
        return []
    if "," in value:
        return [t.strip() for t in value.split(",") if t.strip()]
    # No commas. A phrase would be indexed word by word anyway, since the
    # tokeniser splits terms, so nothing is lost by splitting here.
    return [t for t in value.split() if t.strip()]


async def verify(session, key, source: str, speech: str) -> list[str]:
    """Ask a second model, from a different family, whether the values survived.

    This replaces a rule table built from the detector, which over thirty
    prepared steps produced five false rejections and caught nothing: every
    time it fired the model was right and the table was missing an entry. A
    fixed table does not generalise to notation nobody thought of.

    A verifier that fails to answer is reported as a problem, not waved
    through. Silence is not approval when the thing being approved gets read
    aloud to somebody who cannot check it.
    """
    body = {
        "model": prepare.VERIFIER_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prepare.VERIFY_SYSTEM},
            {"role": "user", "content": prepare.verify_prompt(source, speech)},
        ],
    }
    try:
        async with session.post(
            URL,
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        return prepare.read_verdict(payload["choices"][0]["message"]["content"])
    except Exception as exc:
        return [f"verifier could not be reached: {type(exc).__name__}: {exc}"]


async def one(session, key, step, *, retries: int = 2) -> prepare.Result:
    body = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(step)},
        ],
    }
    last = "no attempt made"
    for attempt in range(retries + 1):
        try:
            async with session.post(
                URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                response.raise_for_status()
                payload = await response.json()
            content = payload["choices"][0]["message"]["content"]
            # An empty content field turns up now and then: the request
            # succeeded, the response is well formed, and the model put nothing
            # in it. Left to json.loads that surfaces as "Expecting value: line
            # 1 column 1", which reads like the model produced malformed JSON
            # and sent three attempts looking for a fault in the step text.
            if not content:
                raise ValueError("the model returned an empty response")
            parsed = json.loads(content)
            result = prepare.check(
                step.id, step.text, str(parsed.get("speech", "")), as_terms(parsed.get("terms"))
            )
            if result.speech.strip():
                result.problems.extend(await verify(session, key, step.text, result.speech))
            return result
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.5 * (attempt + 1))
    result = prepare.Result(step_id=step.id, speech="", terms=())
    result.problems.append(f"request failed: {last}")
    return result


async def run(protocol_id: str, corpus: Path, only: list[str] | None) -> int:
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    # The spoken form is what this script is about to write, so it cannot be
    # required in order to read the protocol.
    proto = protocol_mod.load(corpus, protocol_id, require_spoken=False)
    steps = [s for s in proto.steps if not only or s.id in only]
    if not steps:
        print(f"no steps matching {only}", file=sys.stderr)
        return 2

    started = time.monotonic()
    gate = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:

        async def guarded(step):
            async with gate:
                return await one(session, key, step)

        results = await asyncio.gather(*(guarded(s) for s in steps))

    good = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    for r in bad:
        print(f"REJECTED {r.step_id}")
        for problem in r.problems:
            print(f"         {problem}")
        if r.speech:
            print(f"         got: {r.speech}")

    out = corpus / protocol_id / "spoken.yaml"
    existing = yaml.safe_load(out.read_text()) if out.exists() else {}
    existing = existing or {}
    for r in good:
        existing[r.step_id] = {"speech": r.speech, "terms": list(r.terms)}
    # Written in step order, so a diff reads like the protocol.
    ordered = {s.id: existing[s.id] for s in proto.steps if s.id in existing}
    out.write_text(
        "# Generated by scripts/prepare_protocol.py. Reviewed by eye, committed.\n"
        "# The source of truth for what a step says is parsed.yaml; this is how\n"
        "# it is said.\n"
        + yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100)
    )

    print(
        f"\n{len(good)}/{len(steps)} accepted in {time.monotonic() - started:.0f}s "
        f"-> {out}"
    )
    if bad:
        print(f"{len(bad)} rejected and not written; re-run with --only {' '.join(r.step_id for r in bad)}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("protocol")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--only", nargs="*", help="step ids to redo")
    args = ap.parse_args()
    return asyncio.run(run(args.protocol, args.corpus, args.only))


if __name__ == "__main__":
    raise SystemExit(main())
