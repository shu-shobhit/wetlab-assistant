"""Every tunable constant and every credential, read once from the environment.

Nothing else in the package reads os.environ. Keys live in .env, which is
gitignored; .env.example carries placeholders only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class SettingsError(RuntimeError):
    """A required environment variable is missing or empty."""


@dataclass(frozen=True)
class Settings:
    rime_api_key: str
    openrouter_api_key: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    rime_speaker: str
    llm_model: str
    #: How much the conversation model may think before it answers, as
    #: OpenRouter's `reasoning` parameter. `None` leaves the field out of the
    #: request, so the model keeps its own default. The model is where most of
    #: the response time goes, which is why this is reachable without editing
    #: code.
    #:
    #: Not a boolean, because "off" is not available everywhere. Asked to
    #: disable reasoning, `z-ai/glm-5.3-flash` answers 400, "Reasoning is
    #: mandatory for this endpoint and cannot be disabled". It accepts
    #: `minimal`, and that returns zero reasoning tokens, which is the thing
    #: "off" was wanted for.
    llm_reasoning: str | None
    #: Which OpenRouter providers may serve the model, as a comma separated list
    #: of slugs in preference order, or empty to let OpenRouter route. Routing
    #: balances price against speed, and on these models the cheap hosts are the
    #: slow ones: measured on the real prompt, `deepseek/deepseek-v4-flash`
    #: reaches its first token in 0.84 s on `baidu/fp8` and 1.83 s left to route.
    #: Which host serves it matters more than which model it is.
    llm_provider: str
    stt_model: str
    runs_dir: Path
    corpus_dir: Path
    #: Which protocol is read when the room does not name one. Here rather than
    #: in agent.py because two processes need the same answer: the worker uses
    #: it as its fallback, and the web server marks it as the default in the
    #: picker. They read it from one place or they disagree about it.
    protocol_id: str
    room_name: str
    probe_interval_ms: int
    status_dwell_s: float


def _need(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SettingsError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


#: What LLM_REASONING accepts. `off` asks for no reasoning at all, which some
#: models refuse; the rest are OpenRouter's effort levels.
REASONING_CHOICES = ("off", "minimal", "low", "medium", "high")


def _reasoning(name: str) -> str | None:
    """One of the choices, or None for unset.

    Unset has to stay distinguishable from off, because this is sent to a
    provider with its own default and saying nothing is a different request from
    saying no.
    """
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return None
    if raw not in REASONING_CHOICES:
        raise SettingsError(f"{name} is {raw!r}; it takes one of {', '.join(REASONING_CHOICES)}")
    return raw


def load() -> Settings:
    """Read settings from the environment, with .env as a fallback source."""
    load_dotenv(override=False)
    return Settings(
        rime_api_key=_need("RIME_API_KEY"),
        openrouter_api_key=_need("OPENROUTER_API_KEY"),
        livekit_url=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        livekit_api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
        livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
        rime_speaker=os.environ.get("RIME_SPEAKER", ""),
        # Chosen on two findings from the same three logs, not on latency alone.
        # The same walkthrough was spoken into a live microphone once per model,
        # and the runs are kept under runs/evidence/ (RIME_EVIDENCE.md 2a).
        #
        # It is the fastest of the three: 4 787 ms at the tool-turn median
        # against 5 407 ms for claude-sonnet-4.6 and 11 482 ms for
        # glm-5.3-flash, and 3 137 ms against 3 886 ms on turns answered from
        # the standing context.
        #
        # And it announces an expiring timer, which the model it replaces does
        # not. An announcement is requested right after the assistant has
        # spoken, so the conversation ends with an assistant message, and
        # claude-sonnet-4.6 over claude-on-aws answers 400 to that: "This model
        # does not support assistant message prefill." A timer went off in that
        # run and nothing was said. Timer expiry is the one event the design
        # says outranks step speech, so a model that cannot announce one is
        # failing a stated property rather than being slow.
        llm_model=os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash"),
        llm_reasoning=_reasoning("LLM_REASONING"),
        # Pinned to the hosts the measured run used. Left to route, this model
        # is served from somewhere slower, and which host serves it moves the
        # first token by about a second: more than the gap between the models.
        llm_provider=os.environ.get("LLM_PROVIDER", "baidu/fp8,alibaba/fp8").strip(),
        # A provider/model name for the LiveKit inference gateway, not for
        # OpenRouter. It streams, which is the whole point: the batch recogniser
        # it replaces returned a transcript seconds after the speech ended.
        stt_model=os.environ.get("STT_MODEL", "deepgram/nova-3"),
        runs_dir=Path(os.environ.get("RUNS_DIR", "runs")),
        corpus_dir=Path(os.environ.get("CORPUS_DIR", "data/corpus")),
        protocol_id=os.environ.get("PROTOCOL_ID", "neb_q5_m0492"),
        room_name=os.environ.get("ROOM_NAME", "bench"),
        probe_interval_ms=200,
        # How long a status waits before it is spoken. Set to something large
        # to turn statuses off for the controlled comparison in the evidence.
        status_dwell_s=float(os.environ.get("STATUS_DWELL_S", "0.4")),
    )
