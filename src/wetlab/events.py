"""The append-only run log.

Every number in the evidence tables comes out of this file and nowhere else, so
the demo run and the measured run are summarised by the same script. That puts
two requirements on it that an ordinary logger does not have.

It must survive the process dying. Every line is flushed as it is written,
because a run that crashes half way through is exactly the run worth reading.

It must never be the thing that kills the agent. A payload json cannot encode is
converted rather than raised on: a frozenset of span ids becomes a sorted list, a
Path becomes a string, a dataclass becomes a dict. A logging call is not a place
to discover that `played` was a set.

Two runs must never share a file, because a metric computed over a mixed log is
wrong and wrong quietly. A run directory that already holds a log is refused.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Callable

FILENAME = "events.jsonl"


class LogError(RuntimeError):
    """The run directory is not usable for a new log."""


def revision() -> dict[str, Any]:
    """Which commit produced this run, and whether the tree was clean.

    Every figure in the evidence comes from a run, so a run that does not say
    which code made it cannot be checked against the code later. Five commits
    landed on the day these were measured and none of the runs recorded one, so
    telling them apart afterwards meant reading timestamps.

    Best effort. A repository that is not there, or a git that is not
    installed, is recorded as unknown rather than raised on: a run that
    happens is worth more than a run that refused to start over its own
    provenance.
    """
    import subprocess

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        # A dirty tree means the committed code is not what ran, which matters
        # more than the commit itself when a number is being checked.
        "dirty": None if status is None else bool(status.strip()),
    }


def _jsonable(value: Any) -> Any:
    """Best effort, never raises. The log is evidence, not a serialisation test."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        # Sorted so two runs of the same scenario diff cleanly.
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    return str(value)


class Log:
    """One run, one file, opened once and appended to.

    `header` is written as the first row with `type: "header"`. Put in it
    whatever a reader needs to interpret the rest: protocol id, speaker, render
    mode, model ids, git revision.
    """

    def __init__(
        self,
        run_dir: Path,
        header: dict[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._t0 = clock()
        self._dir = Path(run_dir)
        self._path = self._dir / FILENAME
        if self._path.exists():
            raise LogError(
                f"{self._path} already exists; a run directory holds one run, "
                f"or every metric computed from it mixes two"
            )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._closed = False
        self._dropped_after_close = 0
        self.append("header", **header)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, type: str, **payload: Any) -> dict[str, Any]:
        """Write one row. Returns it, so a caller can hand it straight to the page."""
        record: dict[str, Any] = {
            "type": type,
            "t_ms": round((self._clock() - self._t0) * 1000.0, 3),
            "wall": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
        }
        for key, value in payload.items():
            record[key] = _jsonable(value)
        if self._closed:
            # Shutdown is not instant: tasks the run started are still finishing
            # while the log is being closed, and one of them logging must not
            # raise into whatever it was doing. Losing a row after the end of a
            # run costs nothing; killing a task mid-teardown costs the teardown.
            self._dropped_after_close += 1
            return record
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        return record

    @property
    def dropped_after_close(self) -> int:
        return self._dropped_after_close

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> Log:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()


def read(path: Path) -> list[dict[str, Any]]:
    """Every complete row. A truncated last line is dropped, not raised on."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only the tail of a killed run can be malformed; everything before
            # it was flushed whole and is still evidence.
            continue
    return rows
