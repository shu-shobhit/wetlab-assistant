import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from wetlab import events


@pytest.fixture
def clock():
    """A clock the test drives, so t_ms is exact rather than approximately right."""

    class Clock:
        now = 100.0

        def __call__(self) -> float:
            return self.now

    return Clock()


def test_header_is_the_first_row(tmp_path, clock):
    log = events.Log(tmp_path / "run", {"protocol": "handwritten", "speaker": "alexis"}, clock=clock)
    log.append("step.begin", step_id="s1")
    rows = events.read(log.path)
    assert rows[0]["type"] == "header"
    assert rows[0]["protocol"] == "handwritten" and rows[0]["speaker"] == "alexis"
    assert rows[0]["t_ms"] == 0.0
    assert rows[1]["type"] == "step.begin" and rows[1]["step_id"] == "s1"


def test_t_ms_counts_from_the_run_start_and_never_goes_back(tmp_path, clock):
    log = events.Log(tmp_path / "run", {}, clock=clock)
    clock.now = 100.25
    first = log.append("a")
    clock.now = 101.0
    second = log.append("b")
    assert first["t_ms"] == 250.0
    assert second["t_ms"] == 1000.0
    stamps = [r["t_ms"] for r in events.read(log.path)]
    assert stamps == sorted(stamps)


def test_append_returns_the_record_it_wrote(tmp_path, clock):
    log = events.Log(tmp_path / "run", {}, clock=clock)
    record = log.append("cut", context_id="u1", reason="timer")
    assert record == events.read(log.path)[1]
    assert "wall" in record


def test_every_line_is_flushed_as_it_is_written(tmp_path, clock):
    # The log has to survive the process being killed mid-run, which is how a
    # demo failure actually looks. Read the file without closing the writer.
    log = events.Log(tmp_path / "run", {}, clock=clock)
    log.append("timer.fired", timer_id="s1.timer1")
    assert len(log.path.read_text().splitlines()) == 2


def test_payloads_that_json_cannot_take_are_written_anyway(tmp_path, clock):
    # A logging call must never be the thing that kills the agent. Sets and
    # paths turn up in this log all the time: played is a frozenset, and run
    # directories are Paths.
    @dataclass
    class Verdict:
        ok: bool

    log = events.Log(tmp_path / "run", {}, clock=clock)
    record = log.append(
        "advance.blocked",
        played=frozenset({"s1.t1", "s1.v1"}),
        run_dir=Path("/tmp/run"),
        verdict=Verdict(ok=False),
    )
    assert record["played"] == ["s1.t1", "s1.v1"]  # sorted, so the log diffs cleanly
    assert record["run_dir"] == "/tmp/run"
    assert record["verdict"] == {"ok": False}
    assert events.read(log.path)[1] == record


def test_a_second_run_never_writes_into_the_first_ones_log(tmp_path, clock):
    # Two runs in one file would make every metric computed from it wrong, and
    # wrong quietly. Refuse instead.
    events.Log(tmp_path / "run", {}, clock=clock)
    with pytest.raises(events.LogError, match="already"):
        events.Log(tmp_path / "run", {}, clock=clock)


def test_read_skips_a_truncated_last_line(tmp_path, clock):
    # A killed process leaves a half-written line. The rows before it are still
    # evidence and the metrics script should get them.
    log = events.Log(tmp_path / "run", {}, clock=clock)
    log.append("a")
    with log.path.open("a") as fh:
        fh.write('{"type": "b", "t_m')
    rows = events.read(log.path)
    assert [r["type"] for r in rows] == ["header", "a"]


def test_the_log_is_a_context_manager(tmp_path, clock):
    with events.Log(tmp_path / "run", {}, clock=clock) as log:
        log.append("a")
    assert log.closed
    assert json.loads(log.path.read_text().splitlines()[-1])["type"] == "a"


# --- provenance --------------------------------------------------------------


def test_a_run_records_which_commit_produced_it():
    """Every published figure comes from a log. A log that cannot be tied to a
    commit cannot be checked against the code that made it, and five commits
    landed on the day these numbers were measured."""
    rev = events.revision()
    assert set(rev) == {"commit", "dirty"}
    assert rev["commit"]
    assert rev["dirty"] in (True, False, None)


def test_provenance_never_stops_a_run_from_starting(monkeypatch):
    """A run that happens is worth more than a run that refused over its own
    provenance, so a missing git is recorded rather than raised on."""
    import subprocess

    def no_git(*a, **k):
        raise OSError("git is not installed")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert events.revision() == {"commit": "unknown", "dirty": None}
