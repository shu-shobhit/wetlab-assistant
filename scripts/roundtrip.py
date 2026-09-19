#!/usr/bin/env python3
"""Claim 3.1: does laboratory notation survive being spoken?

Every step in the corpus is synthesised twice by Rime, once as the protocol
writes it and once through the offline preparation, then both are transcribed
by the same recogniser the assistant listens with, and each numeric, unit and
acronym token is scored as recovered or not.

No threshold is set in advance. Inventing one before measuring would be a
number with nothing behind it. Both arms are reported, overall and by token
type, and the numeric tokens are reported alone because a rendering that gets
the words right and the numbers wrong is worse than useless.

The recogniser is not an ear. This measures whether the information survives
synthesis, not whether a person under hood noise finds it easy. That belongs
next to the result and is in the evidence document.

    python scripts/roundtrip.py --out runs/roundtrip
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import wave
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
import websockets
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit import rtc  # noqa: E402
from livekit.agents import inference, stt as stt_base  # noqa: E402

from wetlab import protocol as protocol_mod, scoring  # noqa: E402

WS_HOST = "wss://users-ws.rime.ai"
SR = 22050
#: How much audio goes into one frame pushed at the recogniser. 100 ms is what
#: a real microphone track delivers, so the recogniser sees the same shape of
#: input it sees in the product.
FRAME_MS = 100

#: Frames are pushed at wall-clock speed. Pushing faster was tried and it made
#: the recogniser endpoint in the middle of an utterance: one four-second clip
#: came back as two segments with "microlitres" mangled into "My" at the seam.
#: A streaming recogniser uses arrival timing to decide where speech stops, so
#: feeding it four times too fast is not the same measurement.
FEED_RATE = 1.0

#: How long to wait for another event once the audio has all been pushed. The
#: stream does not end on its own after end_input: the final transcript
#: arrives, then iteration simply blocks, so silence is what marks the end.
IDLE_S = 3.0


def wav_bytes(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(pcm)
    return buffer.getvalue()


async def synthesise(key: str, speaker: str, text: str) -> bytes:
    """One utterance through /ws3, the same endpoint the product speaks on."""
    query = urlencode(
        {
            "speaker": speaker,
            "modelId": "mistv3",
            "lang": "eng",
            "audioFormat": "pcm",
            "samplingRate": SR,
            "segment": "never",
            "pauseBetweenBrackets": "true",
            "phonemizeBetweenBrackets": "true",
        }
    )
    pcm = bytearray()
    async with websockets.connect(
        f"{WS_HOST}/ws3?{query}", additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        await ws.send(json.dumps({"text": text, "contextId": "rt"}))
        await ws.send(json.dumps({"operation": "flush", "contextId": "rt"}))
        async for raw in ws:
            event = json.loads(raw)
            if event.get("type") == "chunk":
                pcm.extend(base64.b64decode(event["data"]))
            elif event.get("type") == "done":
                break
            elif event.get("type") == "error":
                raise RuntimeError(event)
    return bytes(pcm)


async def transcribe(engine, pcm: bytes) -> str:
    """Push audio at the streaming recogniser and collect what it says.

    The gateway recogniser has no one-shot API: `_recognize_impl` raises, so it
    cannot be handed a file. Claim 3.1 says the audio is transcribed by the
    same recogniser the assistant uses, so substituting a one-shot recogniser
    would change what the test measures. This feeds it frames instead.
    """
    stream = engine.stream()
    samples = SR * FRAME_MS // 1000
    step = samples * 2  # 16-bit mono

    async def feed() -> None:
        for start in range(0, len(pcm), step):
            chunk = pcm[start : start + step]
            if len(chunk) < 2:
                break
            stream.push_frame(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=SR,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
            )
            await asyncio.sleep(FRAME_MS / 1000 * FEED_RATE)
        stream.end_input()

    pushing = asyncio.create_task(feed())
    parts: list[str] = []
    iterator = stream.__aiter__()
    try:
        while True:
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=IDLE_S)
            except (asyncio.TimeoutError, StopAsyncIteration):
                break
            if event.type == stt_base.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                parts.append(event.alternatives[0].text)
    finally:
        pushing.cancel()
        await stream.aclose()
    return " ".join(p for p in parts if p).strip()


async def run(corpus: Path, protocols: list[str], out: Path, speaker: str | None, keep: bool) -> int:
    load_dotenv()
    key = os.environ.get("RIME_API_KEY")
    if not key:
        print("RIME_API_KEY is not set", file=sys.stderr)
        return 2

    # Resolved here rather than as an argparse default. The default was
    # evaluated at parse time, before load_dotenv had run, so RIME_SPEAKER was
    # always unset and every run measured the fallback voice while reporting
    # that it had read the setting. Two runs were scored on the wrong speaker
    # before the printed line was checked against .env.
    speaker = speaker or os.environ.get("RIME_SPEAKER")
    if not speaker:
        print(
            "RIME_SPEAKER is not set and no --speaker was given. The evidence has "
            "to name the voice the product speaks with, so this will not guess "
            "one.",
            file=sys.stderr,
        )
        return 2

    model = os.environ.get("STT_MODEL", "deepgram/nova-3")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # The plugin expects a job context to own its http session. Outside the
    # worker there is none, so one is supplied here and its lifetime is this
    # script's.
    async with aiohttp.ClientSession() as http:
        engine = inference.STT(model=model, sample_rate=SR, http_session=http)
        for protocol_id in protocols:
            proto = protocol_mod.load(corpus, protocol_id)
            for step in proto.steps:
                if not scoring.tokens_in(step.text):
                    continue  # nothing in this step for either arm to lose
                arms = {"raw": step.text, "prepared": proto.speech_for(step)}
                for arm, text in arms.items():
                    pcm = await synthesise(key, speaker, text)
                    transcript = await transcribe(engine, pcm)
                    scores = scoring.score(step.text, transcript)
                    rows.append(
                        {
                            "protocol": protocol_id,
                            "step": step.id,
                            "arm": arm,
                            "sent": text,
                            "heard": transcript,
                            "tokens": [
                                {
                                    "source": s.token.source,
                                    "kind": s.token.kind,
                                    "recovered": s.recovered,
                                }
                                for s in scores
                            ],
                        }
                    )
                    if keep:
                        clip = out / "clips" / f"{protocol_id}_{step.id}_{arm}.wav"
                        clip.parent.mkdir(parents=True, exist_ok=True)
                        clip.write_bytes(wav_bytes(pcm))
                    got = sum(s.recovered for s in scores)
                    print(f"  {protocol_id}/{step.id:4s} {arm:8s} {got}/{len(scores)}  {transcript[:60]}")

    report = summarise(rows, model, speaker)
    (out / "roundtrip.json").write_text(json.dumps({"summary": report, "rows": rows}, indent=2) + "\n")
    print_report(report)
    print(f"\nwritten to {out}")
    return 0


def rescore(out: Path, corpus: Path) -> int:
    """Score a finished run again, from the transcripts it already saved.

    Synthesis and recognition are the expensive half and their results are in
    the file. Scoring is a pure function of the step's source text and what
    came back, so a change to the scoring can be applied to a run that has
    already happened without spending anything or, more importantly, without
    changing the audio underneath the numbers.

    That separation is why the scoring bugs found on 7 September could be
    corrected against the same recordings rather than against a fresh run that
    would have differed for unrelated reasons.
    """
    path = out / "roundtrip.json"
    if not path.exists():
        print(f"no run at {path}", file=sys.stderr)
        return 2
    doc = json.loads(path.read_text())
    rows = doc["rows"]

    sources: dict[tuple[str, str], str] = {}
    for protocol_id in sorted({r["protocol"] for r in rows}):
        proto = protocol_mod.load(corpus, protocol_id)
        for step in proto.steps:
            sources[(protocol_id, step.id)] = step.text

    changed = 0
    for row in rows:
        source = sources.get((row["protocol"], row["step"]))
        if source is None:
            print(f"  {row['protocol']}/{row['step']} is no longer in the corpus; left as it was")
            continue
        before = {(t["source"], t["kind"]): t["recovered"] for t in row["tokens"]}
        scores = scoring.score(source, row["heard"])
        row["tokens"] = [
            {"source": s.token.source, "kind": s.token.kind, "recovered": s.recovered}
            for s in scores
        ]
        for s in scores:
            was = before.get((s.token.source, s.token.kind))
            if was is not None and was != s.recovered:
                changed += 1
                verb = "now recovered" if s.recovered else "now lost"
                print(f"  {row['protocol']}/{row['step']:4s} {row['arm']:8s} "
                      f"{s.token.source!r} {verb}")

    report = summarise(rows, doc["summary"]["model"], doc["summary"]["speaker"])
    path.write_text(json.dumps({"summary": report, "rows": rows}, indent=2) + "\n")
    print(f"\n{changed} token verdicts changed")
    print_report(report)
    print(f"\nrewritten {path}")
    return 0


def summarise(rows, model: str, speaker: str) -> dict:
    out: dict = {"model": model, "speaker": speaker, "arms": {}}
    for arm in ("raw", "prepared"):
        scores = [
            scoring.Score(scoring.Token(t["source"], t["kind"], ()), t["recovered"])
            for row in rows
            if row["arm"] == arm
            for t in row["tokens"]
        ]
        out["arms"][arm] = scoring.summarise(scores)
    return out


def print_report(report: dict) -> None:
    print(f"\nrecogniser: {report['model']}   voice: {report['speaker']}")
    kinds = sorted({k for arm in report["arms"].values() for k in arm})
    print(f"\n{'':16s} {'raw':>14s} {'prepared':>14s}")
    for kind in ["all", "numeric_only"] + [k for k in kinds if k not in ("all", "numeric_only")]:
        cells = []
        for arm in ("raw", "prepared"):
            entry = report["arms"][arm].get(kind)
            if not entry or not entry["total"]:
                cells.append("       -")
                continue
            cells.append(f"{scoring.percent(entry):5.1f}% {entry['recovered']:>3}/{entry['total']:<3}")
        print(f"{kind:16s} {cells[0]:>14s} {cells[1]:>14s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--protocols", nargs="*", default=["neb_q5_m0492"])
    ap.add_argument("--out", type=Path, default=Path("runs/roundtrip"))
    # No default. RIME_SPEAKER is read inside run(), after load_dotenv.
    ap.add_argument("--speaker", help="overrides RIME_SPEAKER for one run")
    ap.add_argument("--keep-audio", action="store_true", help="write every clip as a wav")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="score the saved run again from its transcripts, synthesising nothing",
    )
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.out, args.corpus)
    return asyncio.run(
        run(args.corpus, args.protocols, args.out, args.speaker, args.keep_audio)
    )


if __name__ == "__main__":
    raise SystemExit(main())
