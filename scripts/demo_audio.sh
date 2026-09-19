#!/usr/bin/env bash
# Route the demo's audio so a screen recording captures Rime at full quality.
#
#     scripts/demo_audio.sh          set the routing up
#     scripts/demo_audio.sh status   say what is routed where
#     scripts/demo_audio.sh down     put it back
#
# The headset is the only working microphone on this machine, and a Bluetooth
# headset with a live microphone is in HFP, where the sink is 16 kHz mono.
# A recorder taking desktop audio from that sink captures Rime at 16 kHz, worse
# than what Rime sent. So the browser plays into a 48 kHz null sink that the
# recorder captures, and a loopback carries the same audio on to the headset so
# the demo is still audible to the person running it.
#
# The card and the devices are discovered rather than named, so this survives
# the headset getting a different index. Everything here is per-user and lives
# in memory: `down` undoes it and so does a reboot.
set -uo pipefail

SINK=demo_capture
PROFILE=headset-head-unit-msbc

card()   { pactl list short cards   | awk '$2 ~ /^bluez_card/   {print $2; exit}'; }
bt_sink(){ pactl list short sinks   | awk '$2 ~ /^bluez_output/ {print $2; exit}'; }
bt_mic() { pactl list short sources | awk '$2 ~ /^bluez_input/  {print $2; exit}'; }

null_module() { pactl list short modules | awk -v s="sink_name=$SINK"      '$2=="module-null-sink" && index($0,s) {print $1; exit}'; }
loop_module() { pactl list short modules | awk -v s="source=$SINK.monitor" '$2=="module-loopback"  && index($0,s) {print $1; exit}'; }

status() {
  echo "default sink : $(pactl get-default-sink)"
  echo "microphone   : $(bt_mic || true)"
  echo "null sink    : module $(null_module || echo -)"
  echo "loopback     : module $(loop_module || echo -)"
  echo
  echo "sinks:"; pactl list short sinks | sed 's/^/  /'
  echo
  # What is playing and into which sink. The one that matters is the browser:
  # if it is not on demo_capture the recording gets silence.
  echo "playback streams:"
  pactl list sink-inputs \
    | awk '/^Sink Input/{id=$3} /Sink:/{s=$2} /application.name =/{sub(/.*= /,""); print "  " id "  sink " s "  " $0}'
}

case "${1:-up}" in
  status)
    status
    ;;

  down)
    L="$(loop_module)"; [ -n "$L" ] && pactl unload-module "$L" && echo "loopback unloaded"
    N="$(null_module)"; [ -n "$N" ] && pactl unload-module "$N" && echo "null sink unloaded"
    BT="$(bt_sink)"
    [ -n "$BT" ] && pactl set-default-sink "$BT" && echo "default sink back to $BT"
    ;;

  *)
    CARD="$(card)"
    [ -n "$CARD" ] || { echo "no Bluetooth audio card. Connect the headset first."; exit 1; }

    # Idempotent: setting the profile it is already in does nothing.
    pactl set-card-profile "$CARD" "$PROFILE" || exit 1
    sleep 1

    BT="$(bt_sink)"; MIC="$(bt_mic)"
    [ -n "$BT" ] && [ -n "$MIC" ] || { echo "headset sink or microphone missing after the profile switch"; exit 1; }

    [ -n "$(null_module)" ] || pactl load-module module-null-sink \
      sink_name="$SINK" rate=48000 channels=2 \
      sink_properties=device.description=DemoCapture > /dev/null || exit 1

    [ -n "$(loop_module)" ] || pactl load-module module-loopback \
      source="$SINK.monitor" sink="$BT" latency_msec=40 > /dev/null || exit 1

    # Default, so a browser started after this lands on the null sink without a
    # pavucontrol click. A browser already running keeps the sink it had, which
    # is why `status` lists every playback stream and its sink.
    pactl set-default-sink "$SINK"

    echo "recorder desktop audio -> $SINK.monitor   (48 kHz stereo)"
    echo "recorder microphone    -> $MIC"
    echo
    status
    ;;
esac
