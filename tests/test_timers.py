import pytest

from wetlab import timers
from wetlab.protocol import Timer


@pytest.fixture
def clock():
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    return Clock()


@pytest.fixture
def fired():
    return []


@pytest.fixture
def scheduler(clock, fired):
    return timers.Scheduler(fired.append, clock=clock)


def timer(id: str, seconds: float) -> Timer:
    return Timer(id=id, label=f"{id} label", seconds=seconds)


def test_a_timer_fires_when_it_is_due(scheduler, clock, fired):
    scheduler.start([timer("t1", 30)], "s1")
    clock.now = 29.9
    assert scheduler.poll() == []
    clock.now = 30.0
    assert [t.id for t in scheduler.poll()] == ["t1"]
    assert [t.id for t in fired] == ["t1"]


def test_a_fired_timer_does_not_fire_again(scheduler, clock, fired):
    scheduler.start([timer("t1", 5)], "s1")
    clock.now = 10.0
    scheduler.poll()
    scheduler.poll()
    assert len(fired) == 1


def test_two_timers_half_a_second_apart_both_fire_in_order(scheduler, clock, fired):
    # PS stress case 2. Dropping the second because the first was mid
    # announcement is how a wash step gets missed.
    scheduler.start([timer("late", 30.5), timer("early", 30.0)], "s1")
    clock.now = 31.0
    assert [t.id for t in scheduler.poll()] == ["early", "late"]
    assert [t.id for t in fired] == ["early", "late"]


def test_timers_from_different_steps_coexist(scheduler, clock, fired):
    scheduler.start([timer("t1", 10)], "s1")
    clock.now = 5.0
    scheduler.start([timer("t2", 2)], "s2")
    clock.now = 7.0
    assert [t.id for t in scheduler.poll()] == ["t2"]
    clock.now = 10.0
    assert [t.id for t in scheduler.poll()] == ["t1"]


def test_remaining_counts_down(scheduler, clock):
    scheduler.start([timer("t1", 30)], "s1")
    clock.now = 12.0
    assert scheduler.remaining() == [(timer("t1", 30), pytest.approx(18.0))]


def test_remaining_never_goes_negative(scheduler, clock):
    scheduler.start([timer("t1", 5)], "s1")
    clock.now = 9.0
    assert scheduler.remaining()[0][1] == 0.0


def test_a_cancelled_timer_never_fires(scheduler, clock, fired):
    scheduler.start([timer("t1", 5)], "s1")
    assert scheduler.cancel("t1")
    clock.now = 10.0
    assert scheduler.poll() == []
    assert fired == []


def test_cancelling_something_that_is_not_running_says_so(scheduler):
    assert not scheduler.cancel("nothing")


def test_cancel_all_clears_everything(scheduler, clock, fired):
    scheduler.start([timer("t1", 5), timer("t2", 6)], "s1")
    scheduler.cancel_all()
    clock.now = 10.0
    assert scheduler.poll() == []
    assert scheduler.remaining() == []


# --- the loop-driven one ------------------------------------------------------


async def test_the_async_scheduler_fires_on_the_loop(fired):
    scheduler = timers.AsyncScheduler(fired.append)
    scheduler.start([timer("t1", 0.05)], "s1")
    import asyncio

    await asyncio.sleep(0.15)
    assert [t.id for t in fired] == ["t1"]


async def test_cancelling_stops_the_loop_callback(fired):
    scheduler = timers.AsyncScheduler(fired.append)
    scheduler.start([timer("t1", 0.05)], "s1")
    scheduler.cancel("t1")
    import asyncio

    await asyncio.sleep(0.15)
    assert fired == []
