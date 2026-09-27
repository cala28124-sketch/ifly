"""Runs on the Pi: listens for the laptop's warnings and plays a fly's buzz through the USB headset,
in the ear on the side the threat is on (both ears when it's in the middle).

Run from the repo root:  python -m pi.commands          (listen to the laptop)
                         python -m pi.commands --test   (buzz left, right, then both, to check the headset)
Uses aplay (built into Raspberry Pi OS); on a Mac it uses afplay, elsewhere it prints instead.
"""
import array
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import zmq
from config import LAPTOP_IP, COMMAND_PORT, WARN_HOLD, AUDIO_DEVICE, EAR_SIDE, SWAP_EARS

BUZZ_HZ, BUZZ_S, GAP_S = 210, 0.4, 0.02   # wingbeat pitch (a fruit fly beats its wings ~200-220 times a
                                          # second), length of one buzz, pause between buzzes

def make_buzz(ear):
    """Write a stereo fly buzz for one ear ("left", "right") or "both"; return its path. The buzz is the
    wingbeat tone plus its overtones (the raspy texture), with its pitch and loudness wavering a little."""
    rate = 44100
    n = int(rate * BUZZ_S)
    fade = int(rate * 0.03)   # 30 ms fade in/out so it doesn't click
    tone, phase = [], 0.0
    for i in range(n):
        t = i / rate
        phase += 2 * math.pi * (BUZZ_HZ + 12 * math.sin(2 * math.pi * 7 * t) + 5 * math.sin(2 * math.pi * 13.3 * t)) / rate
        shape = sum(math.sin(k * phase) / k for k in range(1, 9))         # buzzy, sawtooth-like
        flutter = 0.8 + 0.2 * math.sin(2 * math.pi * 4.1 * t + 1)         # loudness waver
        tone.append(int(9000 * min(1, i / fade, (n - i) / fade) * flutter * shape))
    left, right = ear in ("left", "both"), ear in ("right", "both")
    samples = array.array("h", (s for t in tone for s in (t * left, t * right)))   # interleaved L, R
    path = pathlib.Path(tempfile.gettempdir()) / f"fly_buzz_{ear}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path

EARS = ("left", "right", "both")
sounds = {ear: make_buzz(ear) for ear in EARS}
if shutil.which("aplay"):
    play = {ear: ["aplay", "-q"] + (["-D", AUDIO_DEVICE] if AUDIO_DEVICE else []) + [str(sounds[ear])] for ear in EARS}
elif shutil.which("afplay"):
    play = {ear: ["afplay", str(sounds[ear])] for ear in EARS}
else:
    play = None
    print("no audio player found; printing instead")

alarm = threading.Event()   # set = buzzing
ear = "both"                # which ear the buzz goes to right now

def ear_for(turn):
    """turn: -1 = threat on the image's left ... +1 = right (from the laptop)."""
    if abs(turn) <= EAR_SIDE:
        return "both"
    image_side = "left" if turn < 0 else "right"
    if SWAP_EARS:   # facing backward, the image's left is the rider's right
        return {"left": "right", "right": "left"}[image_side]
    return image_side

def buzzer():
    """Background thread: buzz over and over while the alarm is set."""
    reported = False
    while True:
        alarm.wait()
        if play:
            result = subprocess.run(play[ear], stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                if not reported:   # say it once, not on every buzz
                    print(f"audio failed: {result.stderr.strip()}\n"
                          "  check the headset with  aplay -l  and set AUDIO_DEVICE in config.py")
                    reported = True
                time.sleep(BUZZ_S)   # failed plays return instantly; keep the buzz rhythm
        else:
            print(f"BUZZ ({ear})")
        time.sleep(GAP_S)

threading.Thread(target=buzzer, daemon=True).start()

def warning(on):
    print(f"WARN ON ({ear} ear)" if on else "warn off")
    alarm.set() if on else alarm.clear()

if "--test" in sys.argv:
    for ear in EARS:   # the buzzer thread reads the global ear
        warning(True)
        time.sleep(1.0)
        warning(False)
        time.sleep(0.5)
    sys.exit()

sub = zmq.Context().socket(zmq.SUB)
sub.connect(f"tcp://{LAPTOP_IP}:{COMMAND_PORT}")
sub.setsockopt(zmq.SUBSCRIBE, b"")   # receive every message
print(f"listening for warnings from {LAPTOP_IP}:{COMMAND_PORT}")

last_warn, is_on = 0.0, False
try:
    while True:
        if sub.poll(100):   # wait up to 100 ms for a command
            command = sub.recv_json()
            if command.get("escape"):
                last_warn = time.time()
                if ear != ear_for(command.get("turn", 0.0)):
                    ear = ear_for(command.get("turn", 0.0))
                    if is_on:
                        print(f"  now {ear} ear")
        # stay on for WARN_HOLD after the last warning, so it doesn't stutter between chunks;
        # this also stops it if the laptop stops sending
        should_be_on = time.time() - last_warn < WARN_HOLD
        if should_be_on != is_on:
            warning(should_be_on)
            is_on = should_be_on
except KeyboardInterrupt:
    warning(False)
