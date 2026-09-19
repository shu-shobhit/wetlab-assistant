"""The pointer.

What matters here is mostly what is absent. The module this replaces could
refuse to advance, and every test of it was about when it said no. There is no
no any more, so these are about the pointer staying inside the protocol and
about the record of what has been read, which is the one piece of state that
still changes an answer.
"""

from pathlib import Path

import pytest

from wetlab import position, protocol

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


@pytest.fixture
def pos():
    return position.Position(protocol.load(CORPUS, "neb_q5_m0492"))


def test_it_starts_on_the_first_step_and_not_started(pos):
    """Joining does not begin the protocol. The assistant greets and waits,
    because reading step one at someone still putting their gloves on is the
    behaviour of something that cannot be talked to."""
    assert pos.index == 0
    assert pos.number == 1
    assert pos.started is False


def test_number_is_what_a_person_says(pos):
    pos.next()
    assert pos.index == 1
    assert pos.number == 2


def test_next_stops_at_the_end_rather_than_running_off(pos):
    for _ in range(pos.total + 5):
        pos.next()
    assert pos.index == pos.total - 1
    assert pos.finished


def test_previous_stops_at_the_beginning(pos):
    for _ in range(3):
        pos.previous()
    assert pos.index == 0


def test_goto_takes_the_number_a_person_says(pos):
    assert pos.goto(4).id == "s4"
    assert pos.number == 4


@pytest.mark.parametrize("asked", [0, -3, 99])
def test_goto_clamps_rather_than_refusing(pos, asked):
    """A request for a step past the end is a mishearing. The useful answer is
    the nearest real step and a sentence saying so, which the tool composes
    from the number it asked for and the total. Raising would make the model
    apologise for an error instead of answering.

    The expected number is read from the protocol rather than written down. It
    was written down, as 14, and re-ingesting the corpus onto seventeen steps
    broke a test about clamping for reasons that had nothing to do with
    clamping.
    """
    pos.goto(asked)
    assert pos.number == (1 if asked < 1 else len(pos.protocol.steps))


def test_read_aloud_records_what_reached_the_listener(pos):
    assert pos.has_been_read("s1") is False
    pos.mark_read("s1")
    assert pos.has_been_read("s1") is True


def test_re_reading_a_step_does_not_record_it_twice(pos):
    pos.mark_read("s1")
    pos.mark_read("s2")
    pos.mark_read("s1")
    # A set with an order, and the order is what "the last thing you told me"
    # means.
    assert pos.read_aloud == ["s2", "s1"]


def test_the_window_is_the_current_step_and_five_before_it(pos):
    pos.goto(9)
    window = pos.window()
    assert [s.id for s in window] == ["s4", "s5", "s6", "s7", "s8", "s9"]


def test_the_window_does_not_run_off_the_front(pos):
    pos.goto(2)
    assert [s.id for s in pos.window()] == ["s1", "s2"]


def test_contents_is_one_line_per_step(pos):
    contents = pos.contents()
    assert len(contents) == pos.total
    assert contents[0][0] == 1
    assert contents[-1][0] == pos.total


def test_a_long_step_is_shortened_in_the_contents(pos):
    """The whole table goes into every request, so a line is opening words
    rather than the step."""
    long_step = max(pos.protocol.steps, key=lambda s: len(s.text))
    line = dict(pos.contents())[long_step.index + 1]
    assert len(line.split()) <= 9
    if len(long_step.text.split()) > 8:
        assert line.endswith("...")
