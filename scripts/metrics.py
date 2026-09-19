#!/usr/bin/env python3
"""Every number in the evidence, computed from one events.jsonl.

The version this replaces computed the version 1 numbers: how often an
uninformed advance was prevented, how many hazard values leaked, whether a
readback was accepted. None of those mechanisms exist any more. These are the
four claims in docs/WETLAB_V2_DESIGN.md section 3, and they are all about
voice.

It fails loudly rather than quietly. The old one hardcoded thirteen event
names, and a renamed event made every lookup return nothing, which printed as
null with a note blaming the absence of a browser. So a broken script was
reported as a headless run, and the one metric whose numerator had been
renamed printed 0.0 percent while its denominator kept counting. A missing
required name is now an error with the name in it.

Anything that genuinely cannot be computed from a given run is still reported
as null with a reason, because a missing row and a zero are different findings.

    python scripts/metrics.py runs/<stamp>/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wetlab import eventnames as E  # noqa: E402
from wetlab import events  # noqa: E402


@dataclass
class Metric:
    name: str
    value: float | None
    unit: str
    n: int = 0
    note: str = ""


@dataclass
class Report:
    metrics: list[Metric] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        self.metrics.append(Metric(*a, **k))

    def as_dict(self) -> dict:
        return {
            m.name: {"value": m.value, "unit": m.unit, "n": m.n, "note": m.note}
            for m in self.metrics
        }


def rows(log: list[dict], kind: str) -> list[dict]:
    return [r for r in log if r.get("type") == kind]


def median(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def percentile(values, p: float) -> float | None:
    """Nearest rank. With a dozen turns, interpolating invents precision."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
    return round(values[index], 1)


def check_names(log: list[dict]) -> None:
    present = {r.get("type") for r in log}
    missing = [n for n in E.REQUIRED if n not in present]
    if missing:
        raise SystemExit(
            f"this log contains no {', '.join(missing)} rows.\n"
            "Either the run ended before the assistant spoke, or an event has "
            "been renamed and this script was not updated with it. It is not "
            "a run without a browser."
        )


def turn_ends(log: list[dict]) -> list[dict]:
    """When the scientist stopped talking, however this run was driven.

    A spoken run has user.state going back to listening. A scripted run has
    script.turn. Both mark the moment after which somebody is waiting.
    """
    scripted = rows(log, E.SCRIPT_TURN)
    if scripted:
        return scripted
    return [r for r in rows(log, E.USER_STATE) if r.get("state") == "listening"]


def windows(log: list[dict]):
    """Each turn paired with the moment the next one starts.

    Every per-turn figure is bounded this way. Without the bound a turn that
    produced nothing was credited with the next turn's audio, which turned a
    missing measurement into a thirty-five second response time.
    """
    turns = turn_ends(log)
    for turn, following in zip(turns, list(turns[1:]) + [None]):
        yield turn, turn["t_ms"], following["t_ms"] if following else float("inf")


#: Growth, as a fraction of realtime, that counts as speech rather than as an
#: idle track. The same constant as `web/probe.js`, for the same reason: an RTP
#: audio track keeps emitting while the agent is silent, at about a tenth of
#: realtime, so the counter moving is not evidence that anything was said.
SPEECH_RATE = 0.5


def speech_onsets(log: list[dict]) -> list[dict]:
    """The first position row of each utterance that is growing at realtime.

    Computed here rather than read from `probe.onset`, because every log written
    before 8 September has an onset that fired on any growth at all. That fired
    one poll after the utterance was created no matter what the agent was doing,
    so the response time built on it was the two hundred millisecond poll
    interval wearing the name of a response time: 413 ms published for a run
    whose true median was 8.7 seconds. `web/probe.js` no longer records it that
    way, and deriving it from the positions means the logs already on disk give
    the right answer without having to be recorded again.

    One onset per utterance. One that never reached the listener produces none,
    which is what keeps it out of the figures rather than crediting it with the
    next utterance's audio.
    """
    onsets: list[dict] = []
    previous: dict[str, dict] = {}
    seen: set[str] = set()
    for row in sorted(rows(log, E.PROBE_POSITION), key=lambda r: r["t_ms"]):
        cid = row.get("context_id")
        before, previous[cid] = previous.get(cid), row
        if before is None or cid in seen:
            continue
        grown = float(row.get("rendered_s", 0.0)) - float(before.get("rendered_s", 0.0))
        elapsed = (row["t_ms"] - before["t_ms"]) / 1000
        if elapsed > 0 and grown >= elapsed * SPEECH_RATE:
            seen.add(cid)
            onsets.append(row)
    return onsets


def first_heard(log: list[dict]):
    """Per turn: how long until the first speech reached the listener, and
    whether a tool ran during it."""
    onsets = sorted(speech_onsets(log), key=lambda r: r["t_ms"])
    # Whether a tool ran is read from the tool calls themselves. It used to be
    # inferred from the kind of utterance heard, which worked only while tools
    # did their own speaking. They no longer speak, so every utterance is a
    # reply and that inference silently put eleven of twelve turns in the
    # no-tool column. Asking the log what was called is both simpler and true
    # regardless of who does the talking.
    calls = rows(log, E.TOOL_CALLED)
    out = []
    for turn, start, limit in windows(log):
        window = [o for o in onsets if start <= o["t_ms"] < limit]
        if not window:
            continue
        used_tool = any(start <= c["t_ms"] < limit for c in calls)
        out.append((turn, window[0]["t_ms"] - start, used_tool))
    return out


def compute(log: list[dict]) -> Report:
    check_names(log)
    report = Report()
    heard = first_heard(log)

    # --- claim 3.2, perceived response time --------------------------------
    # Split by whether a tool ran, because those differ by an entire model
    # round trip and one figure over both hides the thing worth knowing.
    with_tool = [gap for _, gap, used_tool in heard if used_tool]
    without = [gap for _, gap, used_tool in heard if not used_tool]
    report.add("response_ms_tool_median", median(with_tool), "ms", len(with_tool),
               "user finished to first sample heard, turns that called a tool")
    report.add("response_ms_tool_p90", percentile(with_tool, 0.9), "ms", len(with_tool))
    report.add("response_ms_no_tool_median", median(without), "ms", len(without),
               "same, turns answered from the standing context alone")
    report.add("response_ms_no_tool_p90", percentile(without, 0.9), "ms", len(without))

    # Advancing is the most frequent thing anyone says, once per step, and
    # removing the pedal made it cost a full recognition and model round trip.
    # That is a real cost of the design rather than something to leave unsaid.
    # When the step reached the model, not when its audio started. There is no
    # longer an utterance that is a step: the model says them, so a step's
    # arrival is the moment it was handed over.
    steps = rows(log, E.STEP_PRESENTED)
    advances = []
    for turn, start, limit in windows(log):
        text = str(turn.get("text", "")).lower()
        if not any(w in text for w in ("done", "next", "move on", "finished")):
            continue
        after = [s for s in steps if start <= s["t_ms"] < limit]
        if after:
            advances.append(min(after, key=lambda r: r["t_ms"])["t_ms"] - start)
    report.add("advance_ms_median", median(advances), "ms", len(advances),
               "spoken advance to the next step beginning")

    # --- claim 3.3, continuity during tool work ----------------------------
    silences = [gap for _, gap, used_tool in heard if used_tool]
    report.add("tool_silence_ms_max", max(silences) if silences else None, "ms", len(silences),
               "longest gap before anything was heard on a tool turn")
    dropped = rows(log, E.STATUS_DROPPED)
    spoken = [r for r in rows(log, E.SPEECH_START) if r.get("kind") == "status"]
    total = len(dropped) + len(spoken)
    report.add("status_dropped_pct",
               round(100 * len(dropped) / total, 1) if total else None,
               "%", total, "statuses dropped because the answer was ready first")

    # --- claim 3.4, interruption -------------------------------------------
    # From the page's probe, not from the agent's idea of playback. The two are
    # different numbers and only one is what the scientist experienced.
    cuts = rows(log, E.PROBE_CUT)
    onsets = [r for r in rows(log, E.USER_STATE) if r.get("state") == "speaking"]
    stops = []
    for cut in cuts:
        before = [o for o in onsets if o["t_ms"] <= cut["t_ms"]]
        if before:
            stops.append(cut["t_ms"] - max(before, key=lambda r: r["t_ms"])["t_ms"])
    report.add("interrupt_stop_ms_median", median(stops), "ms", len(stops),
               "user speech onset to audio stopping, measured at the listener")

    # --- the design's own claims about itself ------------------------------
    # `presynth_used_pct` was here, and the mechanism it measured is gone.
    # Rime rendered the next step during the pause while the scientist worked,
    # which only worked because a tool spoke the step and so its words were
    # known in advance. The model's words are not, so there is nothing to
    # render early and nothing to measure.
    report.add("steps_presented", len(rows(log, E.STEP_PRESENTED)), "count",
               len(rows(log, E.STEP_PRESENTED)),
               "steps handed to the model to say")

    # Every step is spoken by the model now, so there is no utterance kind that
    # means "a step". What can still be said is whether what the model said
    # reached the listener without being cut.
    replies = [r for r in rows(log, E.SPEECH_FINISHED) if r.get("kind") == "reply"]
    heard = [r for r in replies if not r.get("interrupted")]
    report.add("replies_heard_pct",
               round(100 * len(heard) / len(replies), 1) if replies else None,
               "%", len(replies), "model replies that finished without being cut")

    failures = rows(log, E.PROVIDER_FAILURE)
    report.add("provider_failures", len(failures), "count", len(failures),
               "with no pedal there is no fallback path, so this is reported")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = compute(events.read(args.log))
    width = max(len(m.name) for m in report.metrics)
    for m in report.metrics:
        value = "null" if m.value is None else f"{m.value:g}"
        print(f"{m.name:<{width}}  {value:>8} {m.unit:<5} n={m.n:<4} {m.note}")

    out = args.out or args.log.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    with (out / "results.csv").open("w") as fh:
        fh.write("name,value,unit,n,note\n")
        for m in report.metrics:
            note = m.note.replace('"', "'")
            fh.write(f'{m.name},{"" if m.value is None else m.value},{m.unit},{m.n},"{note}"\n')
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
