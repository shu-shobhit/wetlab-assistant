"""What the model can do, as tools.

Eleven of them. Five speak protocol text, one retrieves without speaking, three
run timers, one reports position and one ends the protocol. None can refuse
anything: the version this replaces had an `advance` that could return "blocked:
step s1 has not been read all the way through", and there is nothing left to
block.

Anything the assistant can say it did has to be here. Ending was not, so asked
to end the protocol it said it had and nothing happened: the pointer stayed
where it was and a twenty minute timer kept counting.

Every tool takes a `status`, and the model fills it when it makes the call. It
is one short sentence saying what the assistant is about to do, in the words of
the question it was asked: "checking the annealing step", not "let me check".
It costs no extra model call, because it arrives in the same response that
invokes the tool, and that is what makes it contextual rather than a phrase
chosen in advance. speech.py drops it if the tool returns before it is due.

A value reaches the listener through the prepared spoken form, spoken by the
tool. What a tool returns is for the model to compose a sentence around.
"""

from __future__ import annotations

from typing import Callable

from livekit.agents import RunContext, ToolError, function_tool


def build_tools(app) -> list[Callable]:
    """Bind the tools to one running assistant. `app` is agent.Assistant."""

    @function_tool
    async def start_protocol(ctx: RunContext, status: str = "") -> str:
        """Begin the protocol. Call this when the scientist asks to start, to
        begin, or to go from the top. Returns the first step, in the wording to
        say it in."""
        async with app.doing(status):
            app.position.started = True
            app.position.goto(1)
            return await app.present_step()

    @function_tool
    async def read_current(ctx: RunContext, status: str = "") -> str:
        """The current step again, in the wording to say it in. Use this when
        the scientist did not catch it, or asks you to repeat or say it
        again."""
        async with app.doing(status):
            return await app.present_step()

    @function_tool
    async def next_step(ctx: RunContext, status: str = "") -> str:
        """Move to the next step. Call this when the scientist says they have
        done the current step, or are ready to move on. Returns the new step,
        in the wording to say it in."""
        async with app.doing(status):
            if app.position.finished:
                return f"already on the last step, {app.position.number} of {app.position.total}"
            app.advance()
            return await app.present_step()

    @function_tool
    async def previous_step(ctx: RunContext, status: str = "") -> str:
        """Go back one step. Returns it, in the wording to say it in."""
        async with app.doing(status):
            app.position.previous()
            return await app.present_step()

    @function_tool
    async def go_to_step(ctx: RunContext, number: int, status: str = "") -> str:
        """Jump to a step by its number. `number` counts from one, as the
        scientist says it. Returns the step, in the wording to say it in."""
        async with app.doing(status):
            total = app.position.total
            app.position.goto(number)
            text = await app.present_step()
            if not 1 <= number <= total:
                # Clamped rather than refused. Which step it went to is more
                # use to the scientist than an apology, and the model is told
                # what happened so it can say so before reading it.
                return (
                    f"there is no step {number}; this protocol has {total}. "
                    f"Went to {app.where()} instead: {text}"
                )
            return text

    @function_tool
    async def look_up(
        ctx: RunContext,
        query: str = "",
        step_numbers: list[int] | None = None,
        status: str = "",
    ) -> str:
        """Find steps in the protocol without reading them aloud.

        Use `query` to search by what the scientist asked about, in their own
        words: "temperature", "how long to anneal", "the master mix". Use
        `step_numbers` to fetch particular steps. Both may be passed together.

        Each result says whether it has already been read aloud, which is the
        difference between what the scientist has heard and what the document
        contains.
        """
        async with app.doing(status):
            hits = app.index.search(
                query=query or None,
                steps=step_numbers or None,
                read_aloud=app.position.read_aloud,
            )
            if not hits:
                return "nothing in this protocol matches that"
            return "\n".join(
                f"step {h.number}"
                f"{' (already read aloud)' if h.read_aloud else ''}: "
                f"{app.spoken_text(h.step_id)}"
                for h in hits
            )

    @function_tool
    async def start_timer(ctx: RunContext, seconds: float, label: str, status: str = "") -> str:
        """Start a timer the scientist asks for. `label` is how it should be
        announced when it expires. A timer a step declares starts on its own
        when the scientist says they have done that step, so this is only for
        one they ask for."""
        async with app.doing(status):
            if seconds <= 0:
                raise ToolError("a timer needs a positive number of seconds")
            app.start_ad_hoc_timer(seconds, label)
            return f"started: {label}, {seconds:.0f} seconds"

    @function_tool
    async def cancel_timer(ctx: RunContext, label: str, status: str = "") -> str:
        """Stop a running timer. `label` is how it was announced."""
        async with app.doing(status):
            return "cancelled" if app.cancel_timer(label) else f"no timer matching {label!r}"

    @function_tool
    async def list_timers(ctx: RunContext, status: str = "") -> str:
        """What is running, and how long is left on each."""
        async with app.doing(status):
            remaining = app.remaining_timers()
            if not remaining:
                return "no timers running"
            return "; ".join(
                f"{timer.label}, {seconds:.0f} seconds left" for timer, seconds in remaining
            )

    @function_tool
    async def finish_protocol(ctx: RunContext, status: str = "") -> str:
        """End the protocol. Call this when the scientist says they are done
        with it, want to stop, or want to end the session, whether or not they
        reached the last step. It stops every running timer. Returns where it
        ended and what was stopped."""
        async with app.doing(status):
            return app.finish()

    @function_tool
    async def where_are_we(ctx: RunContext, status: str = "") -> str:
        """Which step the protocol is on, and what that step says. Only when
        the scientist asks where they are. The tools that move already return
        the step, so calling this straight after one of those asks twice for
        something you have already been given."""
        async with app.doing(status):
            return app.where(with_text=True)

    return [
        start_protocol,
        read_current,
        next_step,
        previous_step,
        go_to_step,
        look_up,
        start_timer,
        cancel_timer,
        list_timers,
        finish_protocol,
        where_are_we,
    ]
