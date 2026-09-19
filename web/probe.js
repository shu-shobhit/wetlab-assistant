// The playback probe: the one measurement the page makes, and the reason it is
// not a dumb terminal.
//
// The framework's playback position is computed at the agent, from frames it has
// handed to the transport. What actually left the speaker is later than that, by
// the network hop and the receiver's jitter buffer. `jitterBufferEmittedCount`
// counts samples the buffer has released to the output device, which is the
// closest thing the browser exposes to "the scientist has heard this".
//
// Nothing gates on it. Architecture 7.4 made played state a single flag on the
// whole utterance, so this measures the gap between the two numbers rather than
// deciding anything with either. That is worth reporting instead of assuming,
// and it is what gate 4 is.
//
// The onset is the first poll whose counter grows at something like realtime.
// The rate test is the whole of it: an RTP audio track keeps emitting while the
// agent is silent, about a tenth of realtime, twenty milliseconds per poll. The
// first version fired the onset on any growth at all, so it fired one poll after
// `speech.start` no matter what the agent was doing, and the response time built
// on it was the poll interval wearing the name of a response time. It published
// 413 ms for a run whose true median was 8.7 seconds, and the log said so
// plainly: onset at 48.79 s, `agent.state: speaking` at 55.87 s, and twenty
// milliseconds of "audio" per poll in between.
//
// The baseline is re-taken at the onset, so `rendered_s` counts speech rather
// than speech plus however much trickle accumulated while the model was
// thinking. That biases it low by up to one poll interval, which is the safe
// direction for a number describing how much the scientist heard.

window.probe = (() => {
  const INTERVAL_MS = 200;
  const DEFAULT_CLOCK_RATE = 48000;
  //: Growth, as a fraction of realtime, that counts as speech rather than as an
  //: idle track. The trickle is around 0.1 and speech is around 1.0, so anything
  //: between them works and the midpoint is not a tuned number.
  const SPEECH_RATE = 0.5;

  let track = null;
  let timer = null;

  let contextId = null;
  let baseline = null; // emitted count when this utterance's speech started
  let onsetSent = false;
  let lastEmitted = null;
  let lastAt = null; // performance.now() of the poll lastEmitted came from
  let lastRendered = 0;
  let lastDelaySum = null;
  let bufferedS = 0;
  let clockRate = DEFAULT_CLOCK_RATE;

  // Mean seconds a sample waited in the jitter buffer, over the samples emitted
  // since the last poll. Not the lifetime average, which flattens out and stops
  // describing the present.
  function bufferDepth(now) {
    if (lastDelaySum === null || now.delaySum === undefined) return bufferedS;
    const samples = now.emitted - lastEmitted;
    const seconds = now.delaySum - lastDelaySum;
    return samples > 0 ? seconds / samples : bufferedS;
  }

  async function read() {
    // `receiver` is not on the typed surface of RemoteAudioTrack, but it is
    // there at runtime, and getReceiverStats() does not include this counter.
    const receiver = track && track.receiver;
    if (!receiver || !receiver.getStats) return null;
    const stats = await receiver.getStats();
    let inbound = null;
    stats.forEach((report) => {
      if (report.type === "inbound-rtp" && report.kind === "audio") inbound = report;
    });
    if (!inbound || inbound.jitterBufferEmittedCount === undefined) return null;
    const codec = inbound.codecId ? stats.get(inbound.codecId) : null;
    return {
      emitted: inbound.jitterBufferEmittedCount,
      rate: (codec && codec.clockRate) || clockRate,
      concealed: inbound.concealedSamples || 0,
      // Cumulative seconds-of-waiting summed over every emitted sample. Divided
      // by the sample count it is the mean time a sample sits in the buffer,
      // which is how much audio the agent has already sent that the scientist
      // has not heard yet. On a cut that is the tail they will still hear.
      delaySum: inbound.jitterBufferDelay,
    };
  }

  async function poll() {
    const now = await read();
    if (!now) return;
    const at = performance.now();
    bufferedS = bufferDepth(now);
    clockRate = now.rate;

    // How much the counter moved, and over how long. The pair is the whole
    // measurement: the counter alone cannot tell speech from an idle track,
    // because an idle track still moves it.
    const previous = lastEmitted;
    const grown = previous === null ? 0 : (now.emitted - previous) / clockRate;
    const elapsed = lastAt === null ? 0 : (at - lastAt) / 1000;
    lastEmitted = now.emitted;
    lastDelaySum = now.delaySum;
    lastAt = at;

    if (contextId === null) return;

    if (!onsetSent) {
      if (elapsed <= 0 || grown < elapsed * SPEECH_RATE) return;
      onsetSent = true;
      // Speech was already flowing during the interval this poll measured, so
      // the count from before it is the closest sample to where it started.
      // The reported time is still this poll's, which puts the onset up to one
      // interval late. Late makes the response time look worse than it was,
      // which is the direction to be wrong in on a claim about being quick.
      baseline = previous;
      bench.send({
        type: "probe.onset",
        context_id: contextId,
        emitted_samples: now.emitted,
        at_ms: at,
      });
    }

    // The counter is one monotonic total for the track, so it cannot separate
    // the tail of the utterance before this one from the head of this one.
    // After a cut the previous utterance is still draining out of the buffer,
    // and those samples land inside this baseline. That inflates rendered_s for
    // the utterance following a cut by roughly the buffer depth, which is
    // reported alongside it rather than corrected for.
    const rendered = Math.max(0, (now.emitted - baseline) / clockRate);
    if (rendered >= lastRendered) {
      lastRendered = rendered;
      bench.send({
        type: "probe.position",
        context_id: contextId,
        rendered_s: Number(rendered.toFixed(4)),
        buffered_s: Number(bufferedS.toFixed(4)),
        concealed_samples: now.concealed,
        at_ms: at,
      });
    }
  }

  function beginUtterance(id) {
    contextId = id;
    baseline = null;
    onsetSent = false;
    lastRendered = 0;
  }

  // The counter keeps growing after an utterance ends: an idle jitter buffer
  // still emits, slowly, forever. Left running the probe fills the run log with
  // hundreds of rows describing silence, and the evidence is buried in them.
  // The agent says when the utterance is over; stop then.
  //
  // Only for the utterance being measured. A reply that never played still ends,
  // and its ending used to switch the probe off for the one speaking in its
  // place: with preemptive generation on there were two reply handles per turn,
  // and twelve of fourteen utterances in a live run were measured not at all.
  function endUtterance(id) {
    if (id && id !== contextId) return;
    contextId = null;
  }

  function cut(reason) {
    if (contextId === null) return;
    bench.send({
      type: "probe.cut",
      context_id: contextId,
      rendered_s: Number(lastRendered.toFixed(4)),
      // Audio already sent that the scientist has not heard yet, and will,
      // after the cut. The size of the window in which a cut is not a silence.
      buffered_s: Number(bufferedS.toFixed(4)),
      reason: reason || "unknown",
      at_ms: performance.now(),
    });
    contextId = null;
  }

  function start(remoteTrack) {
    track = remoteTrack;
    if (timer !== null) return;
    timer = setInterval(() => {
      poll().catch(() => {
        // getStats can reject while a track is being torn down. A missed poll
        // costs one report, and losing the probe would cost every later one.
      });
    }, INTERVAL_MS);
  }

  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    track = null;
    contextId = null;
    // The counters belong to the track. Keeping them across a reconnect would
    // make the first poll of the new one look like a jump of whatever the old
    // one had reached.
    lastEmitted = null;
    lastAt = null;
    lastDelaySum = null;
  }

  function init() {
    bench.on("speech.start", (m) => beginUtterance(m.context_id));
    // The agent no longer cuts its own speech. The scientist talking over it
    // is the only thing that stops it now, and this records how far the audio
    // had actually got when that happened, from the listener's side rather
    // than from the agent's idea of playback.
    bench.on("speech.interrupted", (m) => cut(m.reason));
    bench.on("step.state", (m) => endUtterance(m.context_id)); // playback finished
    bench.on("_disconnected", stop);
  }

  return { start, stop, cut, init, beginUtterance, endUtterance };
})();
