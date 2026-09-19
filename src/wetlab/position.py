"""Where in the protocol we are, and what has been read aloud.

This replaces state.py, which was a state machine with the authority to refuse
an advance until a step had been read to the end and its dangerous values said
back. Both of those guarantees were dropped on 7 September, so what is left is
a pointer and a record. It refuses nothing, and there is no state to be in
beyond which step is current.

The one thing it still tracks that matters to behaviour is which steps have
been read aloud. That is what separates "what temperature did you mention"
from "what temperatures are in this protocol": the first is a question about
what the listener has heard, the second about what the document contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import Protocol, Step

#: How many steps behind the current one are held in the model's context. The
#: window is fixed rather than sized to the protocol, because a hundred-step
#: protocol would otherwise put the whole document in every request.
WINDOW = 5


@dataclass
class Position:
    """The pointer. Nothing here can fail and nothing here says no."""

    protocol: Protocol
    index: int = 0
    #: Ids of steps whose reading reached the listener, in the order they did.
    read_aloud: list[str] = field(default_factory=list)
    #: False until the scientist asks to begin. On joining, the assistant
    #: greets and waits rather than reading step one at someone who is still
    #: putting their gloves on.
    started: bool = False
    #: True once the scientist has said they are done with the protocol. It
    #: refuses nothing either: reading any step clears it, because reading a
    #: step is what going again looks like.
    ended: bool = False

    @property
    def total(self) -> int:
        return len(self.protocol.steps)

    @property
    def current(self) -> Step:
        return self.protocol.steps[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= self.total - 1

    @property
    def number(self) -> int:
        """The step number a person would say, counting from one."""
        return self.index + 1

    def has_been_read(self, step_id: str) -> bool:
        return step_id in self.read_aloud

    def mark_read(self, step_id: str) -> None:
        """Record that a step's reading reached the listener.

        Re-reading a step does not record it twice. The list is a set with an
        order, and the order is what "the last thing you told me" means.
        """
        if step_id in self.read_aloud:
            self.read_aloud.remove(step_id)
        self.read_aloud.append(step_id)

    def goto(self, number: int) -> Step:
        """Move to a step by the number a person says. Clamped, never refused.

        A request for step 40 of 14 is a mishearing or a mistake, and the
        useful answer is the last step plus a sentence saying so, which the
        tool composes from `number` and `total`.
        """
        self.index = max(0, min(number - 1, self.total - 1))
        return self.current

    def next(self) -> Step:
        self.index = min(self.index + 1, self.total - 1)
        return self.current

    def previous(self) -> Step:
        self.index = max(self.index - 1, 0)
        return self.current

    def window(self) -> tuple[Step, ...]:
        """The current step and the five before it, oldest first."""
        start = max(0, self.index - WINDOW)
        return self.protocol.steps[start : self.index + 1]

    def contents(self) -> tuple[tuple[int, str], ...]:
        """One line per step: its number and its opening words.

        The whole table goes into every request. At three hundred steps that is
        roughly 1,800 tokens per turn, which is affordable and is what lets the
        model answer "which step was the one about the primers" without a
        lookup.
        """
        return tuple(
            (i + 1, _opening(step.text)) for i, step in enumerate(self.protocol.steps)
        )


def _opening(text: str, words: int = 8) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + "..."
