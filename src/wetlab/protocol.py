"""Loading a protocol, and the prepared form of each step.

A step is an id, an index, its text and any declared timers. There are no
spans, no hazard classes and no irreversible flag: those existed only for the
guarantees that were dropped on 7 September, where a person marked every
dangerous value by hand so the renderer could find it and the gate could
require it be said back.

Beside the source sits a generated file, spoken.yaml, holding the speakable
form and the index terms for each step. It is produced offline by
scripts/prepare_protocol.py and committed. Keeping it separate from the source
means the protocol text stays exactly as its citation says it reads, and the
derived form shows up in a diff whenever it changes.

The citation comes from the CITATION file rather than from the YAML, so a
protocol cannot be added without saying where it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Rime's own inline markup, which is an instruction to the speech engine
#: rather than anything a reader should see.
_MARKUP = re.compile(r"spell\(([^)]*)\)")


class ProtocolError(RuntimeError):
    """The corpus on disk is malformed."""


def without_markup(text: str) -> str:
    """The prepared text as a reader should see it.

    `spell(Q)` tells Rime to read the letter out. It is not a word, and the
    one place it must never go is into a language model's context: the model
    repeated it in a reply, so Rime was handed "spell Q" as text to say and
    the listener heard the word "spell". Whatever the model is shown, it may
    say, so everything outside the speech path is stripped.
    """
    return _MARKUP.sub(r"\1", text)


@dataclass(frozen=True)
class Timer:
    id: str
    label: str
    seconds: float


@dataclass(frozen=True)
class Spoken:
    """One step as it should be said, and the words that find it."""

    speech: str
    terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step:
    id: str
    index: int
    text: str
    timers: tuple[Timer, ...] = ()

    @property
    def number(self) -> int:
        """The number a person says, counting from one."""
        return self.index + 1


@dataclass(frozen=True)
class Protocol:
    id: str
    title: str
    citation: str
    steps: tuple[Step, ...]
    #: Step id to prepared form. Empty when spoken.yaml has not been generated,
    #: which is not an error: the detector renders the step instead, less well.
    spoken: dict[str, Spoken] = field(default_factory=dict)

    def step(self, number: int) -> Step | None:
        """By the number a person says. None when there is no such step."""
        if 1 <= number <= len(self.steps):
            return self.steps[number - 1]
        return None

    def speech_for(self, step: Step) -> str:
        """What to read aloud.

        There is no fallback. A rule table used to render a step that had no
        prepared form, and it was worse at the job in exactly the cases that
        matter: it read 20-30 mins as negative thirty and a slash as the word
        "slash". Speaking a step badly is not better than refusing to, because
        the listener cannot see the text to know it went wrong.
        """
        prepared = self.spoken.get(step.id)
        if prepared is None or not prepared.speech.strip():
            raise ProtocolError(
                f"{self.id}/{step.id} has no prepared spoken form; "
                "run scripts/prepare_protocol.py"
            )
        return prepared.speech

    def readable_for(self, step: Step) -> str:
        """The same step, for everything that is not the speech engine.

        The model, the page, and any log a person reads get this one. Only
        Rime gets `speech_for`.
        """
        return without_markup(self.speech_for(step))

    def terms_for(self, step: Step) -> tuple[str, ...]:
        """The words this step can be found by, written by the preparation pass.

        These used to be merged with a rule table's guesses. Measured over a
        set of spoken-style questions the table added nothing the model did not
        already cover, and it only ever fired on units somebody had tabulated,
        which is the wrong half of the problem.
        """
        prepared = self.spoken.get(step.id)
        return tuple(dict.fromkeys(t.lower() for t in (prepared.terms if prepared else ())))


@dataclass(frozen=True)
class Summary:
    """One protocol as the picker needs it: enough to choose by, nothing more.

    `id` is the directory name, because that is what `load` takes. It is not
    read out of parsed.yaml, which carries an id of its own that is free to
    disagree with the directory it sits in.
    """

    id: str
    title: str
    steps: int
    citation: str
    #: Whether every step has a prepared spoken form. A protocol that has been
    #: ingested but not prepared is listed and refused rather than hidden: "this
    #: exists and has not been prepared" is a more useful thing to see than a
    #: protocol that is silently not there.
    ready: bool


def available(corpus_dir: Path) -> tuple[Summary, ...]:
    """Every protocol in the corpus, in directory order.

    Nothing here raises. A directory that is half ingested is left out rather
    than failing the listing, because one malformed protocol must not be able to
    stop the other two being offered.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        return ()
    out: list[Summary] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            proto = load(root, entry.name, require_spoken=False)
        except Exception:
            continue
        ready = all(
            (proto.spoken.get(step.id) is not None and proto.spoken[step.id].speech.strip())
            for step in proto.steps
        )
        out.append(
            Summary(
                id=entry.name,
                title=proto.title or entry.name,
                steps=len(proto.steps),
                citation=proto.citation,
                ready=bool(ready),
            )
        )
    return tuple(out)


def load(corpus_dir: Path, protocol_id: str, *, require_spoken: bool = True) -> Protocol:
    """Read data/corpus/<protocol_id>/, its CITATION, and its spoken form.

    A protocol with no prepared spoken form cannot be read aloud, so by default
    loading one is an error here rather than a failure on the first step. The
    build scripts pass require_spoken=False, because ingesting and preparing a
    protocol both have to load it before the spoken form exists.
    """
    root = Path(corpus_dir) / protocol_id
    parsed = root / "parsed.yaml"
    if not parsed.exists():
        raise ProtocolError(f"no parsed.yaml at {parsed}")

    doc = yaml.safe_load(parsed.read_text())
    citation_file = root / "CITATION"
    citation = citation_file.read_text().strip() if citation_file.exists() else ""
    if not citation:
        raise ProtocolError(f"{protocol_id}: CITATION is missing or empty")

    steps: list[Step] = []
    for index, raw in enumerate(doc.get("steps") or []):
        if not str(raw.get("text", "")).strip():
            raise ProtocolError(f"{protocol_id}: step {index + 1} has no text")
        steps.append(
            Step(
                id=raw["id"],
                index=index,
                text=raw["text"],
                timers=tuple(
                    Timer(id=t["id"], label=t["label"], seconds=float(t["seconds"]))
                    for t in (raw.get("timers") or [])
                ),
            )
        )

    if not steps:
        raise ProtocolError(f"{protocol_id}: no steps")

    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise ProtocolError(f"{protocol_id}: duplicate step ids")

    spoken = load_spoken(root / "spoken.yaml", ids)
    if require_spoken:
        missing = [i for i in ids if not spoken.get(i) or not spoken[i].speech.strip()]
        if missing:
            raise ProtocolError(
                f"{protocol_id}: {len(missing)} of {len(ids)} steps have no prepared "
                f"spoken form ({', '.join(missing[:5])}"
                f"{', ...' if len(missing) > 5 else ''}); "
                "run scripts/prepare_protocol.py before reading this protocol aloud"
            )

    return Protocol(
        id=doc["id"],
        title=doc.get("title", ""),
        citation=citation,
        steps=tuple(steps),
        spoken=spoken,
    )


def load_spoken(path: Path, known_ids: list[str]) -> dict[str, Spoken]:
    """The prepared form, if it has been generated.

    A missing file is not an error: the detector can render a step, just less
    well than a model does. A file naming a step that does not exist *is* an
    error, because it means the source was edited and the prepared form was
    not regenerated, and the two would then disagree about what a step says.
    """
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    unknown = set(doc) - set(known_ids)
    if unknown:
        raise ProtocolError(
            f"{path}: names steps that are not in parsed.yaml: {sorted(unknown)}; "
            "regenerate it with scripts/prepare_protocol.py"
        )
    return {
        step_id: Spoken(
            speech=str(entry.get("speech", "")),
            terms=tuple(str(t) for t in (entry.get("terms") or [])),
        )
        for step_id, entry in doc.items()
    }
