"""The event names, in one place.

The version of metrics.py this replaces hardcoded thirteen name strings. When a
name changed, every lookup for it returned an empty list, `statistics.median`
was skipped, and the metric printed as null with a note saying "no probe.cut
rows; run without a browser". So a renamed event was reported as *ran headless*
rather than as *the script is broken*, and the one metric whose numerator was
renamed printed 0.0 percent while its denominator kept counting.

Naming them here does not stop a rename. What it does is let the reader assert
that every name it depends on occurs at least once in the log, and say so
loudly when one does not.
"""

from __future__ import annotations

# The turn.
USER_STATE = "user.state"
AGENT_STATE = "agent.state"
SCRIPT_TURN = "script.turn"

# Speech, from the agent's side.
SPEECH_START = "speech.start"
SPEECH_FINISHED = "speech.finished"
SPEECH_INTERRUPTED = "speech.interrupted"
STATUS_DROPPED = "status.dropped"

#: A step was handed to the model to say. This replaced the pre-synthesis
#: events when tools stopped speaking: there is no longer an utterance that is
#: a step, so this is what records that one came up.
STEP_PRESENTED = "step.presented"

#: The scientist said they were done with the protocol and it was ended, which
#: is a different thing from reaching the last step. Recorded because until
#: there was a tool for it the assistant said it had ended one and had not.
PROTOCOL_FINISHED = "protocol.finished"

#: The model was asked to say something nobody asked it for: a greeting, or a
#: timer expiring. Logged separately from the reply it produces, because a
#: model can decline, and a timer that was asked for and never spoken has to
#: be visible rather than silent.
SPEECH_REQUESTED = "speech.requested"

#: What the recogniser made of a user turn, final results only. No metric reads
#: it. It is in the log because afterwards a wrong answer and a misheard
#: question are indistinguishable without it.
USER_TRANSCRIPT = "user.transcript"

# Speech, from the listener's side. These names are chosen in web/probe.js and
# arrive through the data channel, so they are load-bearing in two files.
PROBE_ONSET = "probe.onset"
PROBE_POSITION = "probe.position"
PROBE_CUT = "probe.cut"

# Tools and timers.
TOOL_CALLED = "tool.called"
TIMERS_STARTED = "timers.started"
TIMER_FIRED = "timer.fired"
TIMER_CANCELLED = "timer.cancelled"

# Failures.
PROVIDER_FAILURE = "provider_failure"

#: Every name a metric reads. Checked against the log before anything is
#: computed, so a rename is reported rather than silently returning null.
REQUIRED = (
    SPEECH_START,
    SPEECH_FINISHED,
    AGENT_STATE,
)

#: Names a metric reads but which a given run may legitimately not contain: a
#: run with no browser has no probe rows, a run where nothing was interrupted
#: has no cut.
OPTIONAL = (
    PROBE_ONSET,
    PROBE_POSITION,
    PROBE_CUT,
    SPEECH_INTERRUPTED,
    STATUS_DROPPED,
    STEP_PRESENTED,
    SPEECH_REQUESTED,
    USER_TRANSCRIPT,
    PROTOCOL_FINISHED,
    TIMERS_STARTED,
    TIMER_FIRED,
    USER_STATE,
    SCRIPT_TURN,
)
