// Connection to the room. Holds no protocol logic: it joins, publishes the mic,
// plays the agent's track, and moves JSON both ways on the "wetlab" topic.
//
// The agent's track is rendered with track.attach(), the normal WebRTC path.
// Do not pull it into Web Audio: the browser's echo cancellation uses the
// rendered output as its reference, and that is what lets the scientist talk
// while the assistant is speaking without the assistant hearing itself.

window.bench = (() => {
  const handlers = {};
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const TOPIC = "wetlab";

  let room = null;
  let agentTrack = null;

  function dispatch(msg) {
    (handlers[msg.type] || []).forEach((fn) => fn(msg));
    (handlers["*"] || []).forEach((fn) => fn(msg));
  }

  // A fresh room per join, always.
  //
  // The server dispatches an agent job when a room is CREATED, not when a
  // participant joins one that already exists. Joining a room the previous
  // session left behind therefore connects fine and then sits in silence, with
  // nothing in any log to say why. That is the shape of the failure after a
  // worker restart or a page reload, which is exactly when it would happen
  // during a demo. Reusing a name buys nothing: the name is only routing.
  //
  // The name also carries the chosen protocol, as `bench.<protocol_id>.<random>`.
  // That is not decoration: dispatch happens at creation, so the agent has to
  // load a protocol before there is a session for this page to send a message
  // to, and the name is the only thing it has at that point. The agent checks
  // the id against the corpus rather than trusting it, and falls back to its
  // default when there is nothing usable here.
  function freshRoomName(prefix, protocolId) {
    const rand = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
    return `${prefix}.${protocolId || ""}.${rand}`;
  }

  async function connect(protocolId, identity = "scientist", prefix = "bench") {
    const roomName = freshRoomName(prefix, protocolId);
    const res = await fetch(
      `/token?room=${encodeURIComponent(roomName)}&identity=${encodeURIComponent(identity)}`
    );
    if (!res.ok) throw new Error(`token request failed: ${res.status}`);
    const { token, url } = await res.json();

    room = new LivekitClient.Room({
      adaptiveStream: false,
      dynacast: false,
      audioCaptureDefaults: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== LivekitClient.Track.Kind.Audio) return;
      agentTrack = track;
      document.body.appendChild(track.attach());
      if (window.probe) window.probe.start(track);
      dispatch({ type: "_track_subscribed" });
    });

    room.on(LivekitClient.RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
      if (topic !== TOPIC) return;
      let msg;
      try {
        msg = JSON.parse(decoder.decode(payload));
      } catch {
        return;
      }
      if (msg && msg.type) dispatch(msg);
    });

    room.on(LivekitClient.RoomEvent.Disconnected, () => dispatch({ type: "_disconnected" }));

    await room.connect(url, token);

    // The microphone is not required to join. Without it the scientist cannot
    // speak or read back, but the assistant still reads steps and the pedal
    // still works. That is the right behaviour when the device is missing,
    // denied, or already held by something else, and it is also what lets the
    // page be driven headlessly.
    let micError = null;
    try {
      await room.localParticipant.setMicrophoneEnabled(true);
    } catch (err) {
      micError = err.message || String(err);
    }

    dispatch({ type: "_connected", mic_error: micError });
    return room;
  }

  async function send(obj) {
    if (!room) return;
    await room.localParticipant.publishData(encoder.encode(JSON.stringify(obj)), {
      reliable: true,
      topic: TOPIC,
    });
  }

  function on(type, fn) {
    (handlers[type] ||= []).push(fn);
  }

  // Browsers block autoplay until the page has been interacted with.
  async function unlockAudio() {
    if (room && !room.canPlaybackAudio) await room.startAudio();
  }

  return {
    connect,
    send,
    on,
    unlockAudio,
    get room() {
      return room;
    },
    get agentTrack() {
      return agentTrack;
    },
  };
})();
