"""The script that turns a run log into the numbers in RIME_EVIDENCE.md.

This had no tests, which is the wrong way round: it decides three of the four
claims, and it had already produced two wrong answers that were caught by
eye rather than by anything automatic. A turn that produced no audio was
credited with the *next* turn's onset and reported a thirty-five second
ninetieth percentile. The tool split keyed on whichever utterance was heard
first, and since the model's reply streams out before a tool's own speech,
every turn landed in the no-tool column and the split measured nothing.

Both are fixed. These pin the fixes, and cover the arithmetic underneath them,
because a metric that is quietly wrong is worse than one that is missing: a
missing number is visibly missing.

Logs are built here rather than loaded, so a case can be constructed that no
real run happens to contain.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import metrics  # noqa: E402

from wetlab import eventnames as E  # noqa: E402


def row(event, t_ms, **fields):
    # Not `kind`: a row's own `kind` field is one of the things being set here,
    # and the two collided.
    return {"type": event, "t_ms": float(t_ms), **fields}


def turn(t_ms, text="what now"):
    return row(E.SCRIPT_TURN, t_ms, text=text)


def spoke(t_ms, context_id, kind, heard_after_ms=100):
    """One utterance: the agent starting it, and the listener hearing it.

    Two position rows, because when speech began is a rate and a rate needs two
    samples. The first is the poll before any speech had arrived and the second
    is the poll that carried it, at realtime, which is the one the onset is
    timed from.
    """
    heard = t_ms + heard_after_ms
    return [
        row(E.SPEECH_START, t_ms, context_id=context_id, kind=kind),
        row(E.PROBE_POSITION, heard - 200, context_id=context_id, rendered_s=0.0),
        row(E.PROBE_POSITION, heard, context_id=context_id, rendered_s=0.2),
    ]


def trickled(t_ms, context_id, kind, polls=20):
    """An utterance the agent created but has not begun to speak.

    The track keeps emitting while it is silent, about twenty milliseconds per
    two hundred millisecond poll. Nothing here was heard, and nothing here may
    be reported as having been heard.
    """
    out = [row(E.SPEECH_START, t_ms, context_id=context_id, kind=kind)]
    for i in range(polls):
        out.append(
            row(E.PROBE_POSITION, t_ms + 200 * (i + 1), context_id=context_id,
                rendered_s=round(0.02 * (i + 1), 4))
        )
    return out


def tool(t_ms, name="next_step"):
    """A tool call, which is what makes a turn a tool turn.

    It used to be inferred from the kind of utterance heard. That only worked
    while tools did their own speaking; once the model became the only voice
    every utterance was a reply and the inference put eleven of twelve turns
    in the wrong column.
    """
    return row(E.TOOL_CALLED, t_ms, name=name, arguments="{}")


def base(*extra):
    """A log with the rows check_names insists on, plus whatever is given.

    The greeting is what supplies them in a real run: it starts and finishes
    before the first question, so every log has one of each. It is a reply
    rather than a step, so it does not disturb the step figures, and it is at
    a negative offset so it precedes the first turn.
    """
    return [
        row(E.AGENT_STATE, 0, state="listening"),
        row(E.SPEECH_START, -1_000, context_id="greeting", kind="greeting"),
        row(E.SPEECH_FINISHED, -900, context_id="greeting", kind="greeting", played=True),
        *extra,
    ]


def value(report, name):
    return report.as_dict()[name]["value"]


def n_of(report, name):
    return report.as_dict()[name]["n"]


# --- the two bugs that were found by eye -------------------------------------


def test_a_turn_that_produced_no_audio_is_not_credited_with_the_next_turns_onset():
    """The bug that reported a p90 of thirty-five seconds.

    Turn one is answered by nothing at all. Turn two is answered in 100 ms.
    Turn one must be absent from the figures, not counted as having waited
    until turn two was answered.
    """
    log = base(
        turn(0),
        turn(30_000),
        tool(30_050),
        *spoke(30_100, "u2", "reply"),
    )
    report = metrics.compute(log)
    assert n_of(report, "response_ms_tool_median") == 1, "turn one measured nothing"
    # Turn two only: its own onset, 200 ms after it. Not 30 seconds.
    assert value(report, "response_ms_tool_median") == 200.0


def test_the_tool_split_looks_at_the_whole_turn_not_the_first_utterance():
    """The model's reply streams out before the tool's own speech, so the first
    thing heard on a tool turn is a reply. Keying on it put every turn in the
    no-tool column."""
    log = base(
        turn(0),
        *spoke(100, "m1", "reply"),
        tool(400),
        *spoke(900, "u1", "reply"),
    )
    report = metrics.compute(log)
    assert n_of(report, "response_ms_tool_median") == 1, "a tool ran, so this is a tool turn"
    assert n_of(report, "response_ms_no_tool_median") == 0
    # Timed from the first thing actually heard, which is the reply.
    assert value(report, "response_ms_tool_median") == 200.0


def test_a_turn_with_no_tool_is_reported_separately():
    log = base(turn(0), *spoke(100, "m1", "reply"))
    report = metrics.compute(log)
    assert n_of(report, "response_ms_no_tool_median") == 1
    assert n_of(report, "response_ms_tool_median") == 0


# --- the bug that published 413 ms for an eight second wait -------------------


def test_an_idle_track_is_not_reported_as_a_response():
    """The third one found by eye, and the worst.

    `probe.onset` fired on any growth at all in the emitted-sample counter, and
    the counter never stops growing: an RTP audio track keeps emitting while the
    agent is silent, at about a tenth of realtime. So the onset landed one poll
    after the utterance was created whatever the agent was doing, and what it
    measured was the poll interval. RIME_EVIDENCE.md published 413 ms for a run
    whose median wait was 8.7 seconds.
    """
    report = metrics.compute(base(turn(0), *trickled(100, "m1", "reply")))
    assert value(report, "response_ms_no_tool_median") is None
    assert n_of(report, "response_ms_no_tool_median") == 0


def test_the_wait_is_timed_to_the_speech_and_not_to_the_silence_before_it():
    """Four seconds of the model thinking, then speech. The figure is four
    seconds, not the first poll of an idle track."""
    log = base(
        turn(0),
        *trickled(100, "m1", "reply", polls=20),  # to t=4100, a tenth of realtime
        row(E.PROBE_POSITION, 4_300, context_id="m1", rendered_s=0.6),  # realtime
    )
    assert value(metrics.compute(log), "response_ms_no_tool_median") == 4300.0


def test_an_onset_recorded_by_the_old_page_is_ignored():
    """The logs on disk carry one that fired on the trickle. Recomputing from
    the positions is what lets those runs be corrected without being recorded
    again, and makes a run read the same whichever page wrote it."""
    log = base(
        turn(0),
        row(E.PROBE_ONSET, 200, context_id="m1"),  # what the old page wrote
        *trickled(100, "m1", "reply", polls=20),
        row(E.PROBE_POSITION, 4_300, context_id="m1", rendered_s=0.6),
    )
    assert value(metrics.compute(log), "response_ms_no_tool_median") == 4300.0


def test_an_advance_is_not_credited_with_a_step_from_a_later_turn():
    """The same unbounded lookback, in the advance metric.

    The user says "done" and nothing happens. Two turns later a step is read
    for another reason. Without a bound, the failed advance is reported as an
    advance that took half a minute, which is the opposite of what happened.
    """
    log = base(
        turn(0, "done, next one"),
        turn(1_000),
        turn(30_000),
        row(E.STEP_PRESENTED, 30_050, step_id="s9", number=9),
        *spoke(30_100, "m9", "reply"),
    )
    report = metrics.compute(log)
    assert n_of(report, "advance_ms_median") == 0
    assert value(report, "advance_ms_median") is None


def test_an_advance_that_did_happen_is_measured():
    log = base(
        turn(0, "I have done that, move on"),
        tool(300),
        row(E.STEP_PRESENTED, 400, step_id="s2", number=2),
        *spoke(500, "m1", "reply"),
        turn(10_000),
    )
    report = metrics.compute(log)
    assert n_of(report, "advance_ms_median") == 1
    # To the agent starting the step, not to the listener hearing it. The two
    # differ by the network, and this figure is about how long the pointer took
    # to move.
    assert value(report, "advance_ms_median") == 400.0


# --- the arithmetic ----------------------------------------------------------


def test_the_percentile_is_a_rank_not_an_interpolation():
    """With a dozen turns, interpolating between two samples invents precision
    that is not in the data. This picks an actual observation: the one at
    round(p * (n - 1)) in sorted order, so the ninetieth of ten values is the
    ninth, not the largest. Pinned here because the figure is quoted and a
    silent change of definition would move it.
    """
    assert metrics.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9
    assert metrics.percentile([1, 2, 3, 4, 5, 6, 7], 0.9) == 6
    assert metrics.percentile([5], 0.9) == 5
    assert metrics.percentile([], 0.9) is None


def test_the_percentile_never_reports_a_value_that_was_not_observed():
    for values in ([1, 100], [1, 2, 3], [4, 4, 900]):
        assert metrics.percentile(values, 0.9) in values


def test_the_median_ignores_missing_values_rather_than_counting_them_as_zero():
    assert metrics.median([100, None, 300]) == 200.0
    assert metrics.median([]) is None
    assert metrics.median([None]) is None


# --- reporting nothing, rather than reporting zero ---------------------------


def test_a_renamed_event_is_an_error_and_says_which_one():
    """The failure this whole module is shaped around. A renamed event made
    every lookup return nothing, which printed as null with a note blaming the
    absence of a browser, so a broken script read as a headless run."""
    log = [row(E.SPEECH_START, 0, context_id="u1", kind="step")]
    with pytest.raises(SystemExit) as caught:
        metrics.compute(log)
    assert E.AGENT_STATE in str(caught.value)


def test_a_metric_with_nothing_to_measure_is_null_and_not_zero():
    """A missing measurement and a measurement of zero are different findings,
    and the difference matters for the claim that is unmeasured."""
    report = metrics.compute(base(turn(0), *spoke(100, "u1", "step")))
    assert value(report, "interrupt_stop_ms_median") is None
    assert n_of(report, "interrupt_stop_ms_median") == 0


def test_an_interruption_is_measured_from_the_listener_not_the_agent():
    """The agent's idea of playback and what left the speaker are different
    numbers, and only one of them is what the scientist experienced."""
    log = base(
        turn(0),
        *spoke(100, "u1", "step"),
        row(E.USER_STATE, 5_000, state="speaking"),
        row(E.PROBE_CUT, 5_180, context_id="u1"),
    )
    report = metrics.compute(log)
    assert value(report, "interrupt_stop_ms_median") == 180.0


# --- the design's claims about itself ----------------------------------------


def test_a_dropped_status_counts_against_the_statuses_that_were_spoken():
    log = base(
        turn(0),
        *spoke(100, "u1", "status"),
        row(E.STATUS_DROPPED, 200),
        row(E.STATUS_DROPPED, 300),
    )
    assert value(metrics.compute(log), "status_dropped_pct") == 66.7


def test_a_reply_that_was_cut_lowers_the_heard_share():
    """Every step is spoken by the model now, so there is no utterance kind
    that means "a step". What can still be measured is whether what the model
    said reached the listener without being cut."""
    log = base(
        turn(0),
        *spoke(100, "m1", "reply"),
        row(E.SPEECH_FINISHED, 900, context_id="m1", kind="reply", interrupted=False),
        row(E.SPEECH_FINISHED, 1_900, context_id="m2", kind="reply", interrupted=True),
        # The greeting is not a reply and must not count either way.
        row(E.SPEECH_FINISHED, 2_900, context_id="g1", kind="greeting", interrupted=True),
    )
    assert value(metrics.compute(log), "replies_heard_pct") == 50.0


def test_steps_presented_counts_what_the_model_was_given_to_say():
    log = base(
        turn(0),
        *spoke(100, "m1", "reply"),
        row(E.STEP_PRESENTED, 90, step_id="s1", number=1),
        row(E.STEP_PRESENTED, 5_000, step_id="s2", number=2),
    )
    assert value(metrics.compute(log), "steps_presented") == 2


def test_a_spoken_run_uses_the_users_own_turn_ends():
    """A scripted run marks turns with script.turn. A spoken one has only the
    recogniser saying the user stopped talking, and both have to work."""
    log = base(
        row(E.USER_STATE, 0, state="listening"),
        tool(100),
        *spoke(250, "m1", "reply"),
    )
    report = metrics.compute(log)
    assert value(report, "response_ms_tool_median") == 350.0
