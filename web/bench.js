// The bench view: choose a protocol, then watch one being read.
//
// The page decides nothing. Everything on screen is a rendering of a message
// the agent sent, which is what makes it replaceable by a tablet at a real hood
// without changing anything in the agent. The one exception is the timer
// countdown, which ticks locally so the seconds move smoothly; the agent keeps
// its own clock and is the one that fires.
//
// There is no pedal and no key. Everything is done by talking, and a key that
// advanced the protocol would be a second way of doing the one thing the whole
// design says is done by voice. The only button here is the one that joins.
//
// What the page still does that nothing else can is measure: the probe reports
// what actually reached the speaker, and that is where the interruption and
// first-audio figures come from.

window.benchView = (() => {
  const el = (id) => document.getElementById(id);

  // How long a fired timer stays on screen before it is cleared. It is not
  // deleted the moment it fires: seeing it reach zero and hearing the assistant
  // say so are the same event, and the screen should still be showing the thing
  // that is being talked about.
  const FIRED_LINGER_MS = 8000;

  let chosen = null;
  let timers = new Map();
  let ticker = null;
  let partialTurn = null; // the scientist's turn still being recognised
  let lastAssistantTurn = null; // where a tool call gets attached

  // --- setup ------------------------------------------------------------------

  async function loadProtocols() {
    const pane = el("picker");
    let body;
    try {
      const res = await fetch("/protocols");
      if (!res.ok) throw new Error(`/protocols returned ${res.status}`);
      body = await res.json();
    } catch (err) {
      pane.innerHTML = "";
      showSetupError(`could not list protocols: ${err.message}`);
      return;
    }

    const list = body.protocols || [];
    if (list.length === 0) {
      pane.innerHTML = '<div class="picker-empty">no protocols in the corpus</div>';
      return;
    }

    // The default first, then whatever the corpus listed. Alphabetical order
    // put the one that is almost always wanted at the bottom.
    const ordered = [...list].sort(
      (a, b) => (b.id === body.default) - (a.id === body.default)
    );
    pane.innerHTML = "";
    ordered.forEach((p) => pane.appendChild(optionFor(p, body.default)));

    // Pre-select the default if it is readable, otherwise the first that is.
    const readable = ordered.filter((p) => p.ready);
    const preferred =
      readable.find((p) => p.id === body.default) || readable[0] || null;
    if (preferred) select(preferred.id);
  }

  function optionFor(p, defaultId) {
    const label = document.createElement("label");
    label.className = p.ready ? "option" : "option unready";

    const input = document.createElement("input");
    input.type = "radio";
    input.name = "protocol";
    input.value = p.id;
    input.disabled = !p.ready;
    input.addEventListener("change", () => select(p.id));
    label.appendChild(input);

    const title = document.createElement("div");
    title.className = "option-title";
    title.textContent = p.title || p.id;
    label.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "option-meta";
    // "not prepared" is shown rather than the protocol being left out. A
    // protocol that has been ingested and not run through the preparation pass
    // cannot be read aloud, and saying so is more use than it being absent.
    meta.textContent = p.ready
      ? `${p.steps} steps${p.id === defaultId ? " · default" : ""}`
      : `${p.steps} steps · not prepared, run scripts/prepare_protocol.py`;
    label.appendChild(meta);

    if (p.citation) {
      // The first line only. A CITATION file is free to run to paragraphs
      // explaining what was reduced and why, and one of them does; the card
      // wants the source, and the rest is a tooltip away.
      const cite = document.createElement("div");
      cite.className = "option-cite";
      cite.textContent = p.citation.split("\n")[0];
      cite.title = p.citation;
      label.appendChild(cite);
    }
    return label;
  }

  function select(id) {
    chosen = id;
    const input = document.querySelector(`#picker input[value="${CSS.escape(id)}"]`);
    if (input) input.checked = true;
    el("join").disabled = false;
  }

  function showSetupError(text) {
    const node = el("setup-error");
    node.textContent = text;
    node.hidden = false;
  }

  async function join() {
    const button = el("join");
    button.disabled = true;
    button.textContent = "opening…";
    el("setup-error").hidden = true;
    try {
      await bench.connect(chosen);
      await bench.unlockAudio();
      el("setup").hidden = true;
      el("bench").hidden = false;
    } catch (err) {
      button.disabled = false;
      button.textContent = "Open protocol";
      showSetupError(`could not open the protocol: ${err.message || err}`);
    }
  }

  // --- the bar ------------------------------------------------------------------

  function setStatus(text, kind) {
    const node = el("status");
    node.textContent = text;
    node.className = `chip chip-status${kind ? ` ${kind}` : ""}`;
  }

  function setBanner(text, kind) {
    const node = el("banner");
    node.textContent = text || "";
    node.className = `banner${kind ? ` ${kind}` : ""}`;
    node.hidden = !text;
  }

  function hello(m) {
    el("protocol-title").textContent = m.title || m.protocol || "";
    el("protocol-title").title = m.citation || "";
    el("voice").textContent = `${m.model || "?"} · ${m.speaker || "?"}`;
    setStatus("live", "live");
  }

  // --- the step -------------------------------------------------------------------

  function renderStep(m) {
    // A step being read is what carrying on looks like, so it takes the card
    // back out of the ended state without anyone having to say so.
    document.querySelector(".step-card").classList.remove("finished");
    el("step-count").textContent = `step ${m.number} of ${m.total}`;
    el("progress-fill").style.width = `${(m.number / Math.max(1, m.total)) * 100}%`;
    el("step-source").textContent = m.source || m.text || "";

    // The prepared spoken form, shown under the step as written. The two lines
    // differ by exactly what the offline pass did to the text, which is
    // otherwise only visible in a YAML file.
    const spoken = (m.text || "").trim();
    const differs = spoken && spoken !== (m.source || "").trim();
    el("step-spoken").textContent = spoken;
    el("step-spoken-wrap").hidden = !differs;
    setBanner("");
  }

  function paintSpeaking() {
    document.querySelector(".step-card").classList.add("speaking");
    el("speaking-flag").textContent = "speaking";
  }

  function paintDone() {
    document.querySelector(".step-card").classList.remove("speaking");
    el("speaking-flag").textContent = "";
  }

  function paintInterrupted() {
    paintDone();
    setBanner("interrupted; it will answer, then ask whether to carry on");
  }

  // The protocol was ended, which is a different thing from reaching the last
  // step. The timers are already gone, each cancelled individually on the way
  // out; this says so on the card rather than leaving a screen that looks like
  // a session still in progress.
  function protocolFinished(m) {
    document.querySelector(".step-card").classList.add("finished");
    el("step-count").textContent = `ended at step ${m.number} of ${m.total}`;
    el("speaking-flag").textContent = "";
    const stopped = m.timers_stopped || 0;
    setBanner(
      `Protocol ended` +
        (stopped ? `, ${stopped} timer${stopped === 1 ? "" : "s"} stopped` : "") +
        `. Say start to go again.`,
      "done"
    );
  }

  // --- timers -----------------------------------------------------------------

  function renderTimers() {
    const pane = el("timers");
    const running = [...timers.values()].filter((t) => !t.fired).length;
    el("timer-count").textContent = running ? `${running} running` : "";

    if (timers.size === 0) {
      pane.innerHTML =
        '<p class="empty">Nothing running. A timer starts when you say a step is done.</p>';
      stopTicker();
      return;
    }

    const now = performance.now();
    pane.innerHTML = "";
    for (const [, timer] of timers) {
      const left = Math.max(0, timer.dueAt - now) / 1000;
      const done = timer.fired || left === 0;

      const row = document.createElement("div");
      row.className = done ? "timer done" : "timer";

      const label = document.createElement("span");
      label.className = "timer-label";
      label.textContent = timer.label;
      row.appendChild(label);

      const value = document.createElement("span");
      value.className = "timer-left";
      value.textContent = done ? "up" : clock(left);
      row.appendChild(value);

      pane.appendChild(row);
    }
  }

  // Minutes and seconds past a minute, seconds below it. A twenty minute
  // incubation shown as "1174s" is a number to do arithmetic on rather than one
  // to read.
  function clock(seconds) {
    const whole = Math.ceil(seconds);
    if (whole < 60) return `${whole}s`;
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }

  function startTicker() {
    if (ticker === null) ticker = setInterval(renderTimers, 250);
  }

  function stopTicker() {
    if (ticker !== null) clearInterval(ticker);
    ticker = null;
  }

  function startTimers(m) {
    (m.timers || []).forEach((t) => {
      timers.set(t.id, {
        label: t.label || t.id,
        dueAt: performance.now() + t.seconds * 1000,
        fired: false,
      });
    });
    renderTimers();
    startTicker();
  }

  function timerFired(m) {
    const timer = timers.get(m.timer_id);
    if (timer) timer.fired = true;
    renderTimers();
    setBanner(`timer up: ${m.label || m.timer_id}`);
    setTimeout(() => {
      timers.delete(m.timer_id);
      renderTimers();
    }, FIRED_LINGER_MS);
  }

  // Cancelled by an advance, or because it was asked for. Without this the row
  // kept counting down for the rest of the session, and the screen and the
  // scheduler disagreed with only the run log knowing which was right.
  function timerCancelled(m) {
    timers.delete(m.timer_id);
    renderTimers();
  }

  // --- the conversation ---------------------------------------------------------

  function turnNode(role, text, partial) {
    const node = document.createElement("div");
    node.className = `turn ${role}${partial ? " partial" : ""}`;

    const who = document.createElement("div");
    who.className = "turn-who";
    who.textContent = role === "scientist" ? "you" : "assistant";
    node.appendChild(who);

    const what = document.createElement("div");
    what.className = "turn-what";
    what.textContent = text;
    node.appendChild(what);
    return node;
  }

  function transcript(m) {
    const pane = el("talk");
    const empty = pane.querySelector(".empty");
    if (empty) empty.remove();

    // A partial is replaced in place rather than appended, so a sentence that
    // was half heard never stays on screen as a record of what was said.
    if (m.role === "scientist" && !m.final) {
      if (partialTurn) {
        partialTurn.querySelector(".turn-what").textContent = m.text;
      } else {
        partialTurn = turnNode("scientist", m.text, true);
        pane.appendChild(partialTurn);
      }
      scroll(pane);
      return;
    }

    if (m.role === "scientist" && partialTurn) {
      partialTurn.className = "turn scientist";
      partialTurn.querySelector(".turn-what").textContent = m.text;
      partialTurn = null;
      scroll(pane);
      return;
    }

    const node = turnNode(m.role, m.text, false);
    pane.appendChild(node);
    if (m.role === "assistant") lastAssistantTurn = node;
    scroll(pane);
  }

  // Which tool the model chose, under the reply it chose it in. It arrives
  // after the reply, because the call is only readable off the handle once the
  // generation has finished.
  function toolCalled(m) {
    if (!lastAssistantTurn || !m.name) return;
    let line = lastAssistantTurn.querySelector(".turn-tool");
    if (!line) {
      line = document.createElement("div");
      line.className = "turn-tool";
      lastAssistantTurn.appendChild(line);
    }
    line.textContent = `${line.textContent ? `${line.textContent}  ` : ""}${m.name}()`;
  }

  function scroll(pane) {
    pane.scrollTop = pane.scrollHeight;
  }

  // --- the log ------------------------------------------------------------------

  // probe.position arrives five times a second and mostly describes silence, and
  // an interim transcript arrives faster than that. Both are on screen already
  // in the form that means something; here they would bury everything else.
  const NOT_ON_SCREEN = new Set(["probe.position", "probe.onset"]);

  function log(m) {
    if (NOT_ON_SCREEN.has(m.type)) return;
    if (m.type === "transcript" && m.final === false) return;

    const pane = el("log");
    const row = document.createElement("div");
    row.className = "log-row";
    if (m.type === "speech.interrupted" || m.type === "provider_failure") {
      row.className += " alert";
    }
    if (m.type === "speech.start" || m.type === "transcript") row.className += " speech";

    const type = document.createElement("span");
    type.className = "log-type";
    type.textContent = m.type;
    row.appendChild(type);

    const rest = Object.fromEntries(Object.entries(m).filter(([k]) => k !== "type"));
    row.appendChild(document.createTextNode(`  ${JSON.stringify(rest)}`));

    pane.appendChild(row);
    pane.scrollTop = pane.scrollHeight;
  }

  // --- wiring -------------------------------------------------------------------

  function init() {
    el("join").addEventListener("click", join);
    loadProtocols();

    bench.on("_connected", (m) => {
      setStatus("live", "live");
      if (m.mic_error) {
        // Without a microphone there is no way to say anything, and saying
        // things is the only input this product has.
        setBanner(`no microphone (${m.mic_error}); nothing will work`, "bad");
      }
      bench.send({ type: "page.ready", at_ms: performance.now(), mic: !m.mic_error });
    });
    bench.on("_disconnected", () => {
      setStatus("disconnected", "lost");
      stopTicker();
    });

    bench.on("hello", hello);
    bench.on("step.begin", renderStep);
    bench.on("speech.start", paintSpeaking);
    bench.on("step.state", paintDone);
    bench.on("speech.interrupted", paintInterrupted);
    bench.on("timers.started", startTimers);
    bench.on("timer.fired", timerFired);
    bench.on("timer.cancelled", timerCancelled);
    bench.on("protocol.finished", protocolFinished);
    bench.on("transcript", transcript);
    bench.on("tool.called", toolCalled);

    bench.on("*", (m) => {
      if (!m.type.startsWith("_")) log(m);
    });
    window.probe.init();
  }

  return { init, setStatus, setBanner };
})();
