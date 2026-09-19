# Evidence

Four claims. For each one: what was claimed, how it was tested, what the result
was, and what the result does not show. Every number names the file it came
from and the command that regenerates it.

Measured 7 and 8 September 2026.

---

## Setup

| | |
|---|---|
| Speech | Rime `mistv3`, speaker `luna`, `lang=eng`, over `wss://users-ws.rime.ai/ws3`, PCM 22 050 Hz mono, `segment=never`, `pauseBetweenBrackets=true`, `phonemizeBetweenBrackets=true` |
| Recognition | `deepgram/nova-3`, streaming, through LiveKit's inference gateway |
| Turn detection | `turn-detector-v1` at `https://agent-gateway.livekit.cloud/v1`, adaptive interruption, local `v1-mini` fallback |
| Conversation model | `deepseek/deepseek-v4-flash` via OpenRouter, pinned to `baidu/fp8,alibaba/fp8`, `temperature=0.2`, no parallel tool calls. Section 2a has the comparison it was chosen on |
| Offline passes | `z-ai/glm-5.3` ingests, `z-ai/glm-5.3-flash` writes the spoken form, `openai/gpt-5-mini` checks it |
| Transport | `livekit-server --dev` on loopback, `livekit-agents` 1.8.0 |
| Machine | One laptop. Media is local. Recognition and turn detection are hosted |

`luna` is set in `.env`. If it is left unset, the LiveKit Rime plugin uses its
own default, `cove`, and the product would speak in a voice nothing here was
measured on. `scripts/roundtrip.py` fails without a speaker instead of choosing
one.

---

## The corpus

| | |
|---|---|
| Cited protocols | 2 |
| Steps | 31: 17 and 14 |
| Reviewed by a bench scientist | no |

NEB's Q5 High-Fidelity 2X Master Mix PCR protocol (M0492), 50 microlitre arm,
and Addgene's bacterial transformation protocol. Each directory holds
`source.md` as fetched, `parsed.yaml` as ingested, `spoken.yaml` as prepared,
and `CITATION` with the URL and access date.

Both were ingested by `scripts/ingest_protocol.py`, which checks that every
number in a step appears in the source document. Both pass.

---

## 1. Pronunciation and controlled delivery

**Claim.** Laboratory notation is spoken correctly. Sent to a speech engine as
written, the same text comes out as something a listener cannot act on. The
losses are units and reagent names; numbers come through.

**Acceptance test.** Set before the first run. Every step is synthesised twice,
once as the protocol writes it and once through the offline preparation. Both go
to Rime, the audio comes back through the same recogniser the assistant listens
with, and each numeric, unit and acronym token is scored as recovered or not.
The prepared arm has to recover a higher share than the raw arm overall, and on
units and acronyms on their own. No threshold was set in advance.

**Procedure.** `scripts/roundtrip.py`. Both arms use the same voice, the same
model and the same recogniser, so the only thing that differs is the wording.
Frames are pushed at wall-clock speed. Feeding them four times faster made the
recogniser endpoint mid-utterance and turned "microlitres" into "My", which was
measuring the harness and not the speech.

**Result.** 107 tokens per arm, over 31 steps of two protocols.

| | raw | prepared |
|---|---|---|
| **all** | **86.9 %** 93/107 | **98.1 %** 105/107 |
| units alone | 75.6 % 31/41 | 95.1 % 39/41 |
| acronyms alone | 60.0 % 6/10 | 100.0 % 10/10 |
| numbers alone | 100.0 % 56/56 | 100.0 % 56/56 |

The test passes. The prepared arm is higher overall, on units and on acronyms.

Both arms recover all 56 numbers. Rime's normaliser reads a digit correctly and
the recogniser writes it back as a digit. All 14 losses in the raw arm are units
and reagent names.

What that sounds like, from `runs/roundtrip/clips/quoted/`. The four strings
are quoted exactly as they appear in `runs/roundtrip/roundtrip.json`:

| | |
|---|---|
| sent, raw | `Add 25 µl of Q5 High-Fidelity 2X Master Mix to the reaction.` |
| heard | Add twenty five **EL** of five **Guatemalan COTSOL's** high fidelity two **Master Mix** to the reaction. |
| sent, prepared | `Add twenty five microlitres of spell(Q) five High-Fidelity two spell(X) Master Mix to the reaction.` |
| heard | Add twenty five **microliters** of **q five** high fidelity two, x master mix to the reaction. |

In the raw arm the volume is twenty five of something the listener cannot
identify, and the reagent has turned into money. `Q` is the sign for the
Guatemalan quetzal, so Rime's normaliser reads `Q5` as an amount of currency.

The prepared arm's two losses are the recogniser's, not the speech: `picograms`
came back as "pico pc", and the trailing sentence of `neb_q5_m0492/s15` was
transcribed as "Ixtus."

**Limitations.**

- The score comes from a recogniser transcribing the synthesised audio. It
  measures whether the information survives synthesis.
- Tokens are scored one at a time. A raw reading of `250-1,000 µl` heard as
  "two hundred fifty one thousand" scores both numbers as recovered even though
  the range is gone. This makes the raw arm look better than it is on numbers,
  which is one reason the claim does not rest on the numeric figure.
- The scorer accepts a unit's singular form. For `s` that is "second", which is
  also an ordinary word. No token in this run turns on that, but a larger
  corpus could.
- Two protocols, 31 steps. With that few, one token moves the result by about
  a percent.

**Command.**

```bash
python scripts/roundtrip.py --protocols neb_q5_m0492 addgene_transformation --out runs/roundtrip
```

### 1a. Words Rime does not know

`POST /oov` over every word in the three protocols returns 15 it does not have,
out of 177. Rime predicts a pronunciation for those instead of failing, so the
failure is silent.

The product does not speak the source text, it speaks the prepared form, so the
question is how many of the 15 are still sent:

| | |
|---|---|
| Spoken as a word, so Rime predicts it | 8 |
| Inside `spell()`, read letter by letter | 3 |
| Never sent, rewritten by the preparation pass | 4 |

The preparation pass removes 7 of the 15 before synthesis without being aimed
at them: `cm`, `kb`, `mins` and `pg` become centimetre, kilobase, minutes and
picograms, and `dntp`, `lb` and `pcr` go inside `spell()`, where nothing is
predicted.

The 8 that remain are carrier words, not values: denature, denaturation,
thermocycler, thermocycling, nuclease, microcentrifuge, pipetting, optionally.
One of them is audibly wrong in the round trip. `nuclease-free water` comes
back as "nucleus free water" in both arms, because Rime's guess at `nuclease`
is close to another word.

The fix would be a phoneme string from `/phonemize`, which needs a recording of
a person saying each word. Those recordings were not made.

**Command.**

```bash
python scripts/build_lexicon.py && python scripts/oov_reach.py
```

Files: `data/oov.json`, `data/oov_reach.json`.

### 1b. The greeting

Every step a listener hears comes from the prepared corpus. The greeting does
not. The model composes it from an instruction that includes the protocol title
as `parsed.yaml` writes it, digits and all, and every model tested copied `Q5`
out of that title unchanged. So each session opened with the protocol's name in
its raw form, and it was heard the way the raw arm above predicts, as an amount
of Guatemalan currency.

The markup could not catch it. It matched `Q` on a word boundary, and the digit
after the `Q` in `Q5` is exactly what that boundary excludes. `_BOUNDARY` in
`src/wetlab/markup.py` now fences a term with letters only, so the `Q` matches
and the digit is pushed clear of the markup:

| | |
|---|---|
| before | `... Using Q5 High-Fidelity 2X Master Mix` |
| after | `... Using spell(Q) 5 High-Fidelity 2 spell(X) Master Mix` |

Nothing in the prepared corpus contains a digit, so this cannot change what any
authored step synthesises to. `tests/test_markup.py` still reproduces every
prepared step in both protocols exactly, and the greeting is a test case in
that file. The fix is covered by tests and has not yet been heard in a live run.

---

## 2. Perceived response time

**Claim.** The assistant answers fast enough to hold a conversation, and the
delay a tool call adds is stated rather than hidden.

**Acceptance test.** From the event log: the median and ninetieth percentile of
the time between the user's turn ending and the first sample of the reply
reaching the listener, split by whether the turn called a tool. The far end
comes from the listener's own probe in the page, not from the agent's idea of
when playback started.

**Procedure.** The same walkthrough of the seventeen-step protocol, spoken into
a microphone once per conversation model, in a live room with real tools,
retrieval, Rime and the log. Three models were run this way. A browser tab
joins, and its probe supplies the timing.

**Result.** The shipped configuration is the first row.

| conversation model | tool turns | median | ninetieth | context-only turns | median | log under `runs/evidence/` |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-flash` on `baidu/fp8`, shipped | 17 | **4 787 ms** | 6 586 ms | 3 | **3 137 ms** | `response_time_deepseek/` |
| `anthropic/claude-sonnet-4.6` on `claude-on-aws` | 16 | 5 407 ms | 7 181 ms | 5 | 3 886 ms | `response_time_live/` |
| `z-ai/glm-5.3-flash`, reasoning `minimal` | 16 | 11 482 ms | 12 733 ms | 3 | 9 682 ms | `response_time_glm/` |

**This claim does not pass.** 4.8 seconds is slow for a conversation, and the
design's own target was under a second.

Most of the delay is the conversation model. A tool turn spends two model round
trips, one to choose the tool and one to compose the reply once it returns.
Recognition is not the problem, at 0.41 to 0.82 s of transcript delay. Rime is
the smaller of the two large terms, and it is itself slower than realtime here,
so the emitter force-flushes to start playback on most turns.

Which host serves the model matters about as much as which model it is. On this
prompt, 1 246 tokens plus eleven tool schemas, `deepseek-v4-flash` on
`baidu/fp8` reaches its first token in 0.84 s, against 2.18 s for
`claude-sonnet-4.6`. That is why the provider is pinned in the settings rather
than left for OpenRouter to route. This first-token measurement is not on disk
in the repository the way the three run logs are.

**Limitations.**

- 19 to 21 turns per run, spoken once by one person. The ninetieth percentile
  is the observed value at `round(0.9 × (n − 1))` in sorted order, not a
  distribution.
- One run per model. The order of the top two is more reliable than the size
  of the gap between them.
- One turn in the `claude-sonnet-4.6` run produced no speech at all, the timer
  announcement. Section 2a has the cause. Its median is computed over the turns
  that did speak.
- Everything is on loopback, so these are the best case, without whatever a
  network would add.
- The onset is reported at the poll that detected speech, so it can be up to
  one poll interval late. That is 200 ms, in the direction that makes the
  assistant look slower.
- The probe counts jitter-buffer samples arriving at the listener, and an idle
  RTP track still trickles samples. So the probe requires growth at close to
  realtime before it calls an utterance started (`web/probe.js`), and
  `scripts/metrics.py` takes the onset from the position rows. An earlier
  version of this figure, 413 ms, came from a probe that did not do this and
  was measuring the poll interval.

**Command.** The spoken runs are made by joining the page and speaking, and the
log lands under `runs/<stamp>/`. To reproduce the method without a microphone,
the scripted driver feeds a fixed question list into a live session as text:

```bash
WETLAB_SCRIPT=data/scenarios/questions.yaml python -m wetlab.agent dev
```

```bash
python scripts/metrics.py runs/<stamp>/events.jsonl
```

### 2a. Which model, and which host

The model and the provider serving it are the two settings with the largest
effect on response time. Both are set from the environment: `LLM_MODEL`,
`LLM_PROVIDER` and `LLM_REASONING`.

Speed was not the only difference between the three. From the same three logs:

| | `deepseek-v4-flash` (shipped) | `claude-sonnet-4.6` | `glm-5.3-flash` |
|---|---|---|---|
| Tool turn median | **4 787 ms** | 5 407 ms | 11 482 ms |
| Longest silence on a tool turn | **6 795 ms** | 7 280 ms | 13 933 ms |
| Timer expiry announced | yes | **no** | yes |
| Tools exercised, of 11 | **11** | 10 | 10 |
| Replies that finished without being cut | 90.9 % of 22 | 91.7 % of 24 | 90.9 % of 22 |
| Provider failures in the run | 2 | 2 | 5 |
| Statuses dropped because the answer was ready first | 100 % | 100 % | 100 % |

The timer row decided it. A timer expiring is the one event the design says
outranks step speech. On `claude-sonnet-4.6` over `claude-on-aws`, one fired at
197.9 s and the request came back HTTP 400 a second later: that model does not
accept a conversation that ends with an assistant message, and a timer
announcement is requested right after the assistant has spoken. Nothing was
said. The other two announced it, 10.6 s and 12.1 s after the timer fired, each
after a Rime websocket dropped and the plugin retried.

The provider failures are otherwise all Rime, `Rime ws closed unexpectedly`, and
every one was followed by speech that played. The HTTP 400 above is the only
failure in these runs that cost the listener an utterance.

Not settled: cost was not compared, neither alternative was run against the
Addgene protocol, and the shipped model is served from `baidu/fp8` and
`alibaba/fp8`, so that is a second place, after recognition, where data leaves
the machine.

---

## 3. Conversation continuity during tool work

**Claim.** When the assistant has to look something up, it says so in words
that fit the question instead of going silent.

**Acceptance test.** On turns that call a tool, the longest continuous silence
between the user finishing and the assistant speaking. A second figure reports
how often a status was dropped because the tool returned before its dwell
expired, which checks that statuses are not padding fast turns.

**Result.** The mechanism never fires, and the silence it was built to cover is
long enough to have needed it.

| | `deepseek-v4-flash` (shipped) | `claude-sonnet-4.6` | `glm-5.3-flash` |
|---|---|---|---|
| Longest silence on a tool turn | 6 795 ms | 7 280 ms | 13 933 ms |
| Statuses dropped because the answer was ready first | 17 of 17 | 16 of 16 | 17 of 17 |

Every status was written by the model and every one was discarded, because
every tool returned inside the 400 ms dwell. The tools are fast: a pointer
move, a BM25 query over thirty-one steps, a timer. None of them touches the
network. The scientist is waiting on the two model round trips either side of
the tool, and the dwell cannot see those because it is timing the tool call.

This is a finding, not a pass. Covering the silence would mean starting the
dwell when the turn ends instead of when the tool is called. That is a design
change and has not been made. A faster model shortens the silence and does not
make the status fire.

**Limitations.**

- The acceptance test asks for a comparison with statuses on and off. That was
  not run. It can be, with `STATUS_DWELL_S=0` against `STATUS_DWELL_S=99` over
  the same question set.
- With no tool slower than the dwell, this says nothing about whether a spoken
  status covers a genuinely slow lookup. The product does not currently have
  one.

**Command.** As claim 2.

---

## 4. Interruption and recovery

**Claim.** When the user talks over the assistant, the audio stops, nothing
from the cut utterance leaks into the next one, and the assistant answers what
was asked instead of resuming.

**What is built.** Adaptive interruption through the gateway, a fenced Rime
stream that drops audio from a cut context instead of letting it arrive inside
the utterance that replaced it, and a prompt rule to answer the interruption
and then ask whether to carry on. `tests/test_tts_rime.py` covers the fence
against a fake `/ws3`.

**Result.** The mechanism works. The latency is not measured.

Five interruptions across the three spoken runs of section 2, all with
`reason: user_speaking`. Each log has the whole chain: `user.state speaking`,
then `speech.interrupted`, then the page's `probe.cut` with how much of the
utterance had been rendered. All five were followed by an answer to the
interrupting question, and none by the assistant resuming where it had been
cut. In the `deepseek-v4-flash` run both were also followed by an offer to
carry on, which is what the prompt asks for.

| run | interruptions | rendered when cut | still buffered |
|---|---|---|---|
| `claude-sonnet-4.6` | 2 | 5.02 s, 7.66 s | 24 ms, 23 ms |
| `deepseek-v4-flash` | 2 | 2.20 s, 2.24 s | 19 ms, 19 ms |
| `glm-5.3-flash` | 1 | 5.34 s | 29 ms |

The last column is the audio still queued at the listener when the cut was
recorded. 19 to 29 ms is under one frame, so the fence is dropping the rest of
the utterance as intended.

**Limitations.**

- The acceptance test asks for the time from the onset of the user's speech to
  the audio stopping, taken from the listener's probe. That number does not
  exist. `interrupt_stop_ms` pairs each `probe.cut` with the most recent
  `user.state speaking` row, but the agent sends `speech.interrupted` to the
  page from the same handler that logs that row, and the page answers with
  `probe.cut` on receipt. So it measures the round trip of the agent's own
  message over the data channel, on loopback: 2.9 ms, 3.0 ms and 3.0 ms across
  the three runs. That is not an interruption latency and is not reported as
  one.
- Measuring it properly means the probe keeps polling after a cut until
  `jitterBufferEmittedCount` stops growing, and reports when the growth
  stopped. That is a change to `web/probe.js` and to `scripts/metrics.py`, and
  it has not been made.

---

## What none of this shows

- **No bench scientist has reviewed the corpus.** A non-specialist checked the
  steps against their sources, and the ingestion checker only verifies that
  numbers appear in the source, not that the steps are the right steps.
- **Interruption has no latency figure.** The mechanism is shown in the logs.
  The timing is not.
- **Rime does not know 15 of the corpus's 177 words.** `data/oov.json` names
  them. The fix needs recordings that were not made.
- **The greeting fix has not been heard live.** It is covered by tests only.
- **A timer announcement depends on the model accepting a conversation that
  ends with an assistant message.** `claude-sonnet-4.6` on `claude-on-aws` did
  not, and a timer expired silently on it. Nothing in the code prevents the
  next model having its own version of this. The announcement path is only
  exercised by running it.
- **Two protocols, on loopback, on one laptop.**
